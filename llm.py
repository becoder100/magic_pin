"""
llm.py — tiny, provider-agnostic LLM wrapper (standard library only).

Design choices (worth defending in the interview):
- Stdlib `urllib` instead of vendor SDKs: fewer dependencies, one code path,
  and every provider is just "POST some JSON, read some JSON".
- Never retry the same model. The judge gives us ~15s per request. On a
  timeout or error we return None immediately and the caller falls back to a
  deterministic template. A decent message on time beats a great one late.
- Rate limits (Groq free tier: 8k tokens/min PER MODEL, and the requested
  max_tokens is reserved up front):
    * keep max_tokens small (~700) so each call reserves less budget;
    * on a 429 the model goes into cooldown until its retry-after time, and
      the SAME call fails over to a second model with its own budget (a 429
      returns instantly, so this costs no real time);
    * when every model is cooling down we skip the call entirely -> template.
- JSON mode + temperature 0 so outputs are parseable and repeatable.
- Config is read from env vars at call time (not import time).
"""

import json
import logging
import os
import re
import threading
import time
from urllib import error as urlerror
from urllib import request as urlrequest

log = logging.getLogger("llm")

DEFAULT_MODELS = {
    # llama-3.3-70b-versatile was retired from Groq; gpt-oss-120b measured ~1s
    # per message and follows format instructions well.
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}
# Second model used when the first is rate-limited (separate per-model quota on Groq).
DEFAULT_FALLBACK_MODELS = {"groq": "qwen/qwen3.8-27b"}

# OpenAI-compatible chat-completions endpoints (same request/response shape).
OPENAI_COMPATIBLE_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

# Outcome of the most recent LLM call ("not_called", "ok", "http_401", "timeout", ...).
# Shown in /v1/metadata so you can confirm the key actually works, not just that it is set.
last_status = "not_called"

# model -> time.monotonic() until which that model is rate-limited (skipped).
_cooldown_until: dict[str, float] = {}
_state_lock = threading.Lock()


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def model_name() -> str:
    return os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(provider(), "")


def fallback_model_name() -> str:
    """LLM_FALLBACK_MODEL env var; 'none' disables failover."""
    v = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    if v.lower() == "none":
        return ""
    return v or DEFAULT_FALLBACK_MODELS.get(provider(), "")


def models() -> list[str]:
    return [m for m in dict.fromkeys([model_name(), fallback_model_name()]) if m]


def default_timeout() -> float:
    try:
        return float(os.getenv("LLM_TIMEOUT", "9"))
    except ValueError:
        return 9.0


def enabled() -> bool:
    """True only if a key is configured and the provider is known.
    When False, the bot runs in template-only mode (still fully functional)."""
    return bool(os.getenv("LLM_API_KEY", "").strip()) and provider() in DEFAULT_MODELS


def complete_json(system: str, user: str, timeout: float | None = None,
                  max_tokens: int = 700) -> dict | None:
    """Ask the LLM for a JSON object. Returns the parsed dict, or None on ANY failure.

    Callers must treat None as "use the template fallback".
    Note: Groq/OpenAI JSON mode requires the word "JSON" to appear in the
    prompt — the system prompt passed in must mention it.
    """
    global last_status
    if not enabled():
        return None
    start = time.monotonic()
    timeout = timeout or default_timeout()

    for model in models():
        with _state_lock:
            if time.monotonic() < _cooldown_until.get(model, 0.0):
                continue  # this model is rate-limited right now: try the next one
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0.5:  # not enough budget left to be worth trying
            break
        p = provider()
        try:
            if p in OPENAI_COMPATIBLE_URLS:
                text = _call_openai_compatible(p, model, system, user, remaining, max_tokens)
            elif p == "gemini":
                text = _call_gemini(model, system, user, remaining, max_tokens)
            elif p == "anthropic":
                text = _call_anthropic(model, system, user, remaining, max_tokens)
            else:
                return None
            parsed = _parse_json_object(text)
            last_status = "ok" if parsed is not None else "bad_json"
            return parsed
        except urlerror.HTTPError as e:
            last_status = f"http_{e.code}"
            if e.code == 429:
                with _state_lock:
                    _cooldown_until[model] = time.monotonic() + _retry_after_seconds(e)
                log.warning("LLM 429 on %s - cooling down, trying next model", model)
                continue  # fail over to the next model
            log.warning("LLM HTTP %s on %s - falling back to template", e.code, model)
            return None
        except Exception as e:  # timeout, network error, bad response shape...
            last_status = type(e).__name__
            log.warning("LLM call failed on %s (%s: %s) - falling back to template", model, type(e).__name__, e)
            return None
    if last_status.startswith("http_429") or all(
            time.monotonic() < _cooldown_until.get(m, 0.0) for m in models()):
        last_status = "rate_limited"
    return None


def _retry_after_seconds(e: urlerror.HTTPError) -> float:
    """Seconds to pause after a 429, from the Retry-After header (default 10s, capped at 60s)."""
    try:
        return min(60.0, max(1.0, float(e.headers.get("retry-after", 10))))
    except (TypeError, ValueError):
        return 10.0


def probe() -> None:
    """One tiny call to check the key works; the result lands in last_status."""
    complete_json("Reply with a JSON object only.", 'Return JSON {"ok": true}', timeout=8, max_tokens=100)


# --------------------------------------------------------------------------
# Provider-specific request builders
# --------------------------------------------------------------------------

def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        # Explicit User-Agent: Groq's Cloudflare front blocks urllib's default
        # "Python-urllib/x.y" with HTTP 403 (error 1010).
        headers={"Content-Type": "application/json", "User-Agent": "vera-bot/1.0", **headers},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_openai_compatible(p: str, model: str, system: str, user: str,
                            timeout: float, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        # Groq reserves max_tokens against the per-minute quota up front, so keep it
        # tight; ~700 still leaves room for gpt-oss's short "reasoning" tokens.
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if "gpt-oss" in model:
        # Reasoning model: low effort keeps latency ~1s; a short message doesn't need more.
        payload["reasoning_effort"] = "low"
    data = _post(
        OPENAI_COMPATIBLE_URLS[p],
        payload,
        {"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '').strip()}"},
        timeout,
    )
    return data["choices"][0]["message"]["content"]


def _call_gemini(model: str, system: str, user: str, timeout: float, max_tokens: int) -> str:
    key = os.getenv("LLM_API_KEY", "").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _post(
        url,
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"},
        },
        {"x-goog-api-key": key},  # header, not query string, so the key never lands in URL logs
        timeout,
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(model: str, system: str, user: str, timeout: float, max_tokens: int) -> str:
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        {"x-api-key": os.getenv("LLM_API_KEY", "").strip(),
         "anthropic-version": "2023-06-01"},
        timeout,
    )
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _parse_json_object(text: str) -> dict | None:
    """Parse the model's reply as a JSON object. Tolerates code fences, stray
    text, or a <think> block around the object (some models add them in JSON mode)."""
    if not text:
        return None
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None
