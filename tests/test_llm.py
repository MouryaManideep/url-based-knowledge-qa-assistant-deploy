import unittest

from rag.llm import COOLDOWN_S, GeminiClient, LLMError, error_code, reset_cooldowns


class APIError(Exception):
    def __init__(self, code, message="error"):
        super().__init__(message)
        self.code = code


class Chunk:
    def __init__(self, text):
        self.text = text


class Models:
    """Scripted stand-in for client.models: a list of per-call outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def _next(self, model):
        self.calls.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def generate_content_stream(self, model, contents, config):
        outcome = self._next(model)
        for item in outcome:
            if isinstance(item, Exception):
                raise item
            yield Chunk(item)

    def generate_content(self, model, contents, config):
        text = self._next(model)

        class Response:
            pass

        response = Response()
        response.text = text
        return response


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def client_with(outcomes, models=("primary", "backup"), clock=None, events=None, budget_s=150):
    fake = type("Client", (), {})()
    fake.models = Models(outcomes)
    sleeps = []
    clock = clock or Clock()
    llm = GeminiClient("key", list(models), client=fake, sleep=sleeps.append, clock=clock,
                       on_event=(events.append if events is not None else None), budget_s=budget_s)
    return llm, fake.models, sleeps


class GeminiClientTests(unittest.TestCase):
    def setUp(self):
        reset_cooldowns()

    def test_retries_transient_error_then_succeeds(self):
        llm, models, sleeps = client_with([APIError(503), ["Hello ", "[1]"]])
        self.assertEqual("".join(llm.stream("sys", [])), "Hello [1]")
        self.assertEqual(models.calls, ["primary", "primary"])
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(llm.last_model, "primary")

    def test_falls_back_to_next_model_when_overloaded(self):
        llm, models, _ = client_with([APIError(503), APIError(503), ["From backup"]])
        self.assertEqual("".join(llm.stream("sys", [])), "From backup")
        self.assertEqual(models.calls, ["primary", "primary", "backup"])
        self.assertEqual(llm.last_model, "backup")

    def test_quota_and_missing_model_skip_straight_to_next(self):
        for code in (429, 404):
            reset_cooldowns()
            llm, models, sleeps = client_with([APIError(code), ["ok"]])
            self.assertEqual("".join(llm.stream("sys", [])), "ok")
            self.assertEqual(models.calls, ["primary", "backup"])
            self.assertEqual(sleeps, [])

    def test_bad_key_is_not_retried(self):
        llm, models, _ = client_with([APIError(403, "API key not valid")])
        with self.assertRaises(LLMError) as ctx:
            list(llm.stream("sys", []))
        self.assertIn("key", str(ctx.exception))
        self.assertEqual(models.calls, ["primary"])

    def test_all_models_overloaded(self):
        llm, models, _ = client_with([APIError(503)] * 4)
        with self.assertRaises(LLMError) as ctx:
            list(llm.stream("sys", []))
        self.assertIn("overloaded", str(ctx.exception))
        self.assertEqual(len(models.calls), 4)

    def test_error_after_text_started_is_not_retried(self):
        llm, models, _ = client_with([["Partial ", APIError(503)]])
        with self.assertRaises(LLMError) as ctx:
            list(llm.stream("sys", []))
        self.assertIn("interrupted", str(ctx.exception))
        self.assertEqual(models.calls, ["primary"])

    def test_single_model_name_gets_configured_fallbacks(self):
        from config import GEMINI_FALLBACK_MODELS

        llm = GeminiClient("key", "my-model", client=object())
        self.assertEqual(llm.models, ["my-model", *[m for m in GEMINI_FALLBACK_MODELS if m != "my-model"]])

    def test_overloaded_model_is_skipped_for_a_while(self):
        clock = Clock()
        llm, models, _ = client_with([APIError(503), APIError(503), ["first"]], clock=clock)
        self.assertEqual("".join(llm.stream("sys", [])), "first")
        # Next request starts with the healthy backup instead of re-trying the overloaded primary.
        llm, models, _ = client_with([["second"]], clock=clock)
        self.assertEqual("".join(llm.stream("sys", [])), "second")
        self.assertEqual(models.calls, ["backup"])
        # After the cooldown, the primary is first again.
        clock.now += COOLDOWN_S + 1
        llm, models, _ = client_with([["third"]], clock=clock)
        list(llm.stream("sys", []))
        self.assertEqual(models.calls, ["primary"])

    def test_time_budget_stops_the_chain(self):
        clock = Clock()

        class SlowModels(Models):
            def _next(self, model):
                clock.now += 100  # Each failed call "takes" 100 seconds.
                return super()._next(model)

        fake = type("Client", (), {})()
        fake.models = SlowModels([APIError(503)] * 4)
        llm = GeminiClient("key", ["primary", "backup"], client=fake, sleep=lambda s: None, clock=clock, budget_s=150)
        with self.assertRaises(LLMError) as ctx:
            list(llm.stream("sys", []))
        self.assertIn("too long", str(ctx.exception))
        self.assertEqual(len(fake.models.calls), 2)

    def test_timeouts_count_as_transient(self):
        class ReadTimeout(Exception):
            pass

        self.assertEqual(error_code(ReadTimeout("slow")), 504)
        self.assertEqual(error_code(TimeoutError()), 504)
        llm, models, _ = client_with([ReadTimeout("slow"), ["ok"]])
        self.assertEqual("".join(llm.stream("sys", [])), "ok")
        self.assertEqual(models.calls, ["primary", "primary"])

    def test_progress_events_describe_what_is_happening(self):
        events = []
        llm, _, _ = client_with([APIError(503), APIError(503), ["ok"]], events=events)
        list(llm.stream("sys", []))
        self.assertEqual(events[0], "Asking primary…")
        self.assertTrue(any("retrying" in e for e in events))
        self.assertIn("Trying backup…", events)

    def test_generate_json_falls_back(self):
        llm, models, _ = client_with([APIError(503), APIError(503), '{"overview": "x"}'])
        self.assertEqual(llm.generate_json("sys", "prompt"), {"overview": "x"})
        self.assertEqual(models.calls, ["primary", "primary", "backup"])


if __name__ == "__main__":
    unittest.main()
