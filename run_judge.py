"""
run_judge.py — runs magicpin's judge_simulator.py WITHOUT editing it.

Why this exists:
- Groq sits behind Cloudflare, which blocks urllib's default
  "Python-urllib/x.y" User-Agent (HTTP 403, error 1010). judge_simulator.py
  uses that default, so its Groq calls fail. We install a global urllib opener
  with a normal User-Agent before the simulator runs.
- It fills the simulator's CONFIG block from our .env (LLM_PROVIDER /
  LLM_API_KEY / LLM_MODEL), so the key never has to be pasted into
  judge_simulator.py.

Usage:
    python run_judge.py                 # warmup
    python run_judge.py all             # any scenario: warmup, phase2_short,
                                        # auto_reply_hell, intent_transition,
                                        # hostile, all, full_evaluation
    python run_judge.py all http://my-bot.example.com   # different bot URL
"""

import os
import sys
import urllib.request

from bot import _load_dotenv  # same .env loader the bot uses

_load_dotenv()

# 1) Every urllib request in this process now sends a normal User-Agent.
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", "magicpin-judge-sim/1.0")]
urllib.request.install_opener(_opener)

import judge_simulator as sim  # noqa: E402  (import AFTER the opener is installed)

# 2) Fill the simulator's CONFIG from args / .env (module globals are read at call time).
sim.TEST_SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "warmup"
# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and adds ~2s per call.
sim.BOT_URL = sys.argv[2] if len(sys.argv) > 2 else os.getenv("BOT_URL", "http://127.0.0.1:8080")
sim.LLM_PROVIDER = os.getenv("JUDGE_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "groq")
sim.LLM_API_KEY = os.getenv("JUDGE_LLM_API_KEY") or os.getenv("LLM_API_KEY", "")
# The simulator's own Groq default (llama-3.1-70b-versatile) is retired. On Groq each
# model has its own per-minute token quota, so the judge uses a DIFFERENT model than
# the bot (gpt-oss-120b, falling back to qwen) and doesn't eat the bot's budget.
sim.LLM_MODEL = os.getenv("JUDGE_LLM_MODEL") or (
    "openai/gpt-oss-20b" if sim.LLM_PROVIDER == "groq" else os.getenv("LLM_MODEL", ""))

# 3) The judge shares our Groq key (free tier: ~8k tokens/min) with the bot. On a
#    429 the simulator silently substitutes a fake "5/10 — could not evaluate"
#    score, which makes results meaningless. For local testing only, make the
#    JUDGE wait and retry on 429 (the bot itself never retries).
import time as _time  # noqa: E402
from urllib import error as _uerr  # noqa: E402


def _with_rate_limit_retry(complete):
    def wrapped(self, prompt, system=None):
        for _attempt in range(8):
            try:
                return complete(self, prompt, system)
            except _uerr.HTTPError as e:
                if e.code != 429:
                    raise
                wait = float(e.headers.get("retry-after", 10) or 10)
                print(f"    (judge rate-limited by provider; waiting {wait:.0f}s)")
                _time.sleep(min(60.0, wait + 1))
        return complete(self, prompt, system)
    return wrapped


for _cls in (sim.OpenAIProvider, sim.AnthropicProvider, sim.GeminiProvider, sim.DeepSeekProvider,
             sim.GroqProvider, sim.OpenRouterProvider):
    _cls.complete = _with_rate_limit_retry(_cls.complete)

if __name__ == "__main__":
    sim.main()
