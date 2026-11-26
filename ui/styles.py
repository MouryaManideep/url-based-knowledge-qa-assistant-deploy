CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 6rem; max-width: 1080px;}
.hero {display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
  padding:1.4rem 1.6rem; border-radius:18px; margin-bottom:1rem; border:1px solid #E6E8F0;
  background:linear-gradient(135deg,#EEF2FF 0%,#F5F3FF 55%,#ECFEFF 100%);}
.hero h1 {font-size:1.8rem; line-height:1.2; margin:0; padding:0; color:#111827; font-weight:700; letter-spacing:-0.02em;}
.hero p {margin:.35rem 0 0; color:#4B5563; font-size:1rem;}
.stats {display:flex; gap:.5rem; flex-wrap:wrap;}
.stats span {background:#fff; border:1px solid #E5E7EB; border-radius:999px; padding:.3rem .8rem; font-size:.82rem; color:#374151; white-space:nowrap;}
.stats b {color:#4F46E5;}
.steps {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.9rem; margin:1.2rem 0 1rem;}
@media (max-width: 820px) {.steps {grid-template-columns:1fr;}}
.step {border:1px solid #E6E8F0; border-radius:16px; padding:1.1rem 1.2rem; background:#fff;}
.step .num {display:inline-flex; width:1.7rem; height:1.7rem; border-radius:50%; align-items:center; justify-content:center;
  background:#EEF2FF; color:#4F46E5; font-weight:700; font-size:.9rem; margin-bottom:.5rem;}
.step h4 {margin:0 0 .25rem; padding:0; font-size:1rem; color:#111827;}
.step p {margin:0; color:#6B7280; font-size:.9rem;}
.brand {font-weight:700; font-size:1.15rem; color:#111827; margin:-.5rem 0 .25rem; letter-spacing:-0.01em;}
.brand small {display:block; font-weight:400; color:#6B7280; font-size:.8rem; margin-top:.1rem;}
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button,
[data-testid="stColumn"] .stButton button, [data-testid="column"] .stButton button,
[data-testid="stColumn"] .stDownloadButton button, [data-testid="column"] .stDownloadButton button {width:100%;}
[data-testid="stSidebar"] h4 {margin-top:.6rem;}
[data-testid="stChatMessage"] {border-radius:14px;}
div[role="radiogroup"] {gap:.25rem;}
</style>
"""

WELCOME = """
<div class="steps">
  <div class="step"><div class="num">1</div><h4>Add sources</h4>
    <p>Paste web links or upload PDFs, scans, photos of documents and text files.</p></div>
  <div class="step"><div class="num">2</div><h4>Ask questions</h4>
    <p>Hybrid semantic and keyword search finds the most relevant passages.</p></div>
  <div class="step"><div class="num">3</div><h4>Check the citations</h4>
    <p>Every answer links back to the exact passages it came from.</p></div>
</div>
"""
