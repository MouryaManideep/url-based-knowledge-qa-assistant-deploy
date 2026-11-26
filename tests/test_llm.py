import unittest

from rag.llm import GeminiClient, LLMError


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


def client_with(outcomes, models=("primary", "backup")):
    fake = type("Client", (), {})()
    fake.models = Models(outcomes)
    sleeps = []
    return GeminiClient("key", list(models), client=fake, sleep=sleeps.append), fake.models, sleeps


class GeminiClientTests(unittest.TestCase):
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

    def test_generate_json_falls_back(self):
        llm, models, _ = client_with([APIError(503), APIError(503), '{"overview": "x"}'])
        self.assertEqual(llm.generate_json("sys", "prompt"), {"overview": "x"})
        self.assertEqual(models.calls, ["primary", "primary", "backup"])


if __name__ == "__main__":
    unittest.main()
