"""
bot.py — the HTTP API the magicpin judge talks to (FastAPI).

Endpoints: POST /v1/context, POST /v1/tick, POST /v1/reply,
           GET /v1/healthz, GET /v1/metadata, POST /v1/teardown (optional).

Key design choices:
- All state is in memory (dicts guarded by one lock). The brief allows this;
  it means we must run exactly ONE uvicorn worker and never restart mid-test.
- The bot never reads dataset/ files: everything arrives via /v1/context.
- "Rules decide, AI writes": this file only decides WHICH triggers to act on
  (deterministic rules). composer.py decides WHAT to say.
- Hard time budget on /v1/tick: messages are composed in parallel; anything not
  finished by the deadline gets the instant template fallback, so we always
  answer inside the judge's 15s timeout.
- The lock is never held while composing or calling the LLM, so /v1/healthz
  and /v1/context stay fast even while a slow tick is running.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (avoids a python-dotenv dependency).
    Real environment variables win over the file, so hosting-platform secrets
    always take priority."""
    full = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(full):
        return
    with open(full, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

import composer  # noqa: E402  (imported after .env so llm config is visible)
import conversation_handlers  # noqa: E402
import facts  # noqa: E402
import llm  # noqa: E402

# The brief (§7.1) asks bot.py to expose compose(category, merchant, trigger, customer).
# The implementation lives in composer.py; this re-export is the submission entry point.
compose = composer.compose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bot")

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """On startup, probe the LLM once in a background thread so /v1/metadata can
    report whether the key really works. Background = startup and healthz never wait on it."""
    log.info("LLM provider=%s model=%s key_loaded=%s",
             llm.provider(), llm.model_name(), llm.enabled())
    if llm.enabled():
        threading.Thread(target=llm.probe, daemon=True).start()
    yield


app = FastAPI(title="Vera bot — magicpin AI challenge", lifespan=_lifespan)
START = time.time()

VALID_SCOPES = ("category", "merchant", "customer", "trigger")
TICK_BUDGET_SECONDS = 11.0   # simulator times out tick at 15s; keep a safety margin
REPLY_BUDGET_SECONDS = 11.0  # same for /v1/reply
MAX_ACTIONS_PER_TICK = 20    # hard cap from the testing brief

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_lock = threading.RLock()
contexts: dict[tuple[str, str], dict] = {}   # (scope, context_id) -> {"version", "payload"}
conversations: dict[str, dict] = {}          # conversation_id -> conversation record
sent_suppression_keys: set[str] = set()      # dedup: never send the same key twice
sent_trigger_ids: set[str] = set()           # never act on the same trigger twice
opted_out_merchants: set[str] = set()        # said stop / hostile -> no proactive sends
auto_reply_counts: dict[str, int] = {}       # merchant_id -> consecutive auto-replies seen
_conv_seq = 0                                # makes conversation_ids unique

# Shared worker pool for composing messages in parallel (LLM calls are I/O-bound).
_executor = ThreadPoolExecutor(max_workers=8)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get(scope: str, context_id: str | None) -> dict | None:
    """Return the latest payload for a context, or None."""
    if not context_id:
        return None
    with _lock:
        entry = contexts.get((scope, context_id))
        return entry["payload"] if entry else None


# ---------------------------------------------------------------------------
# Malformed requests -> 400 (the spec says 400, FastAPI's default would be 422)
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={
        "accepted": False, "reason": "malformed_request", "details": str(exc.errors())[:500]})


# ---------------------------------------------------------------------------
# GET /v1/healthz and /v1/metadata
# ---------------------------------------------------------------------------
@app.get("/v1/healthz")
async def healthz():
    counts = {scope: 0 for scope in VALID_SCOPES}
    with _lock:
        for (scope, _cid) in contexts:
            counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Rohit Kumar",
        "team_members": ["Rohit Kumar"],
        "model": (" -> ".join(llm.models()) + " (fallback: per-kind templates)")
                 if llm.enabled() else "templates-only (no LLM key set)",
        # "ok" = key works; "http_401" = bad key; "http_429" = rate-limited; "not_called" = no call yet.
        "llm_status": llm.last_status if llm.enabled() else "disabled",
        "approach": ("Rules decide, AI writes: deterministic trigger selection + per-kind "
                     "fact sheet of verified numbers; LLM (temp 0, JSON mode) writes the message; "
                     "validator rejects invented numbers/taboos/URLs; per-kind template fallback. "
                     "Rule-first reply classifier for auto-reply, intent, opt-out and off-topic."),
        "contact_email": "rohitkumar_23it137@dtu.ac.in",
        "version": "0.2.0",
        "submitted_at": os.getenv("SUBMITTED_AT", "2026-09-27T00:00:00Z"),
    }


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------
class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {list(VALID_SCOPES)}"})

    key = (body.scope, body.context_id)
    with _lock:
        current = contexts.get(key)
        # Idempotent: same or older version is a no-op, reported as 409.
        if current and current["version"] >= body.version:
            return JSONResponse(status_code=409, content={
                "accepted": False, "reason": "stale_version",
                "current_version": current["version"]})
        # Higher version replaces the old one in a single assignment (atomic under the lock).
        # facts.clean repairs cp1252-garbled text ("â‚¹299" -> "₹299") once, at the door.
        contexts[key] = {"version": body.version, "payload": facts.clean(body.payload)}

    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": _now_iso()}


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------
class TickBody(BaseModel):
    now: str = ""
    available_triggers: list[str] = []


def _select_candidates(trigger_ids: list[str]) -> list[dict]:
    """Deterministic rules for WHICH triggers to act on this tick.

    Note: we deliberately do NOT drop triggers whose expires_at is in the past.
    The seed triggers expire around May 2026 while the simulator uses the real
    clock, so strict expiry would make the bot send nothing.
    """
    eligible = []
    for tid in dict.fromkeys(trigger_ids):  # de-duplicate, keep order
        trg = _get("trigger", tid)
        if not trg:
            continue  # trigger never pushed to us
        merchant_id = trg.get("merchant_id") or (trg.get("payload") or {}).get("merchant_id")
        merchant = _get("merchant", merchant_id)
        category = _get("category", (merchant or {}).get("category_slug"))
        if not (merchant and category):
            continue  # can't write a grounded message without these

        sup_key = trg.get("suppression_key") or f"{trg.get('kind')}:{tid}"
        with _lock:
            if merchant_id in opted_out_merchants:
                continue  # they told us to stop — never message proactively again
            if sup_key in sent_suppression_keys or tid in sent_trigger_ids:
                continue  # already sent this one

        customer_id = trg.get("customer_id")
        customer = _get("customer", customer_id)
        if trg.get("scope") == "customer" and customer:
            # Consent: respect an explicit opt-out of reminders.
            if (customer.get("preferences") or {}).get("reminder_opt_in") is False:
                continue
        # Customer-facing trigger without a CustomerContext is still allowed:
        # the composer writes a nameless, detail-free version.

        eligible.append({"trigger_id": tid, "trigger": trg, "merchant_id": merchant_id,
                         "merchant": merchant, "category": category,
                         "customer_id": customer_id, "customer": customer,
                         "suppression_key": sup_key})

    # Highest urgency first; trigger id as a stable tie-breaker (determinism).
    eligible.sort(key=lambda c: (-int(c["trigger"].get("urgency") or 0), c["trigger_id"]))

    # At most ONE merchant-facing message per merchant per tick (don't spam);
    # customer-facing messages are limited to one per customer.
    chosen, seen_merchants, seen_customers = [], set(), set()
    for c in eligible:
        if c["trigger"].get("scope") == "customer":
            key = c["customer_id"] or c["trigger_id"]
            if key in seen_customers:
                continue
            seen_customers.add(key)
        else:
            if c["merchant_id"] in seen_merchants:
                continue
            seen_merchants.add(c["merchant_id"])
        chosen.append(c)
    return chosen[:MAX_ACTIONS_PER_TICK]


def _clean_outgoing(body: str, vera: bool) -> str:
    """Final pass on every first message: Unicode spaces -> real spaces (no glued
    words), no double signs, a/an fixed; Vera's Hindi first person is feminine."""
    body = composer.normalize(body)
    return facts.feminine_vera(body) if vera else body


def _new_conversation_id(candidate: dict) -> str:
    """Readable + unique, e.g. conv_m_001_drmeera_research_digest_003."""
    global _conv_seq
    with _lock:
        _conv_seq += 1
        seq = _conv_seq
    who = candidate["customer_id"] or candidate["merchant_id"] or "unknown"
    short = "_".join(str(who).split("_")[:3])
    kind = candidate["trigger"].get("kind", "msg")
    return f"conv_{short}_{kind}_{seq:03d}"


@app.post("/v1/tick")
def tick(body: TickBody):
    # Plain `def` (not async): FastAPI runs it in a thread, so blocking LLM
    # calls here never freeze /v1/healthz.
    deadline = time.monotonic() + TICK_BUDGET_SECONDS
    candidates = _select_candidates(body.available_triggers)

    futures = {
        _executor.submit(composer.compose, c["category"], c["merchant"], c["trigger"],
                         c["customer"], body.now, deadline): c
        for c in candidates
    }
    if futures:
        wait(futures, timeout=max(0.0, deadline - time.monotonic()))

    actions = []
    for fut, c in futures.items():
        msg = None
        if fut.done():
            try:
                msg = fut.result()
            except Exception as e:
                log.warning("compose failed for %s: %s", c["trigger_id"], e)
        else:
            fut.cancel()  # too slow — the template below replaces it
        if not msg or not str(msg.get("body", "")).strip():
            msg = composer.fallback_compose(c["category"], c["merchant"], c["trigger"],
                                            c["customer"], body.now)

        conv_id = _new_conversation_id(c)
        action = {
            "conversation_id": conv_id,
            "merchant_id": c["merchant_id"],
            "customer_id": c["customer_id"],
            "send_as": msg.get("send_as") or ("merchant_on_behalf" if c["trigger"].get("scope") == "customer" else "vera"),
            "trigger_id": c["trigger_id"],
            "template_name": msg.get("template_name") or "vera_generic_v1",
            "template_params": msg.get("template_params") or [],
            "body": _clean_outgoing(msg["body"], vera=(msg.get("send_as") or "vera") == "vera"),
            "cta": msg.get("cta") or "open_ended",
            "suppression_key": c["suppression_key"],
            "rationale": msg.get("rationale") or "Composed from category + merchant + trigger context.",
        }
        actions.append(action)

        with _lock:
            sent_suppression_keys.add(c["suppression_key"])
            sent_trigger_ids.add(c["trigger_id"])
            conversations[conv_id] = {
                "conversation_id": conv_id,
                "merchant_id": c["merchant_id"],
                "customer_id": c["customer_id"],
                "trigger_id": c["trigger_id"],
                "send_as": action["send_as"],
                "status": "open",
                "messages": [{"from": "bot", "body": action["body"], "cta": action["cta"],
                              "ts": body.now or _now_iso()}],
            }

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str = "merchant"
    message: str = ""
    received_at: str = ""
    turn_number: int = 0


def _safe_reply(result: Any, conv: dict) -> dict:
    """Final safety net: whatever the handler returned, emit a valid response.
    Malformed or empty sends cost -2 each, and a repeated body costs -2, so
    both become a harmless 'wait' instead."""
    if not isinstance(result, dict) or result.get("action") not in ("send", "wait", "end"):
        return {"action": "wait", "wait_seconds": 3600,
                "rationale": "Could not form a confident reply; backing off 1h."}
    if result["action"] == "send":
        text = composer.normalize(str(result.get("body") or ""))
        already_sent = {m["body"] for m in conv["messages"] if m["from"] == "bot"}
        if not text or text in already_sent:
            return {"action": "wait", "wait_seconds": 3600,
                    "rationale": "Avoided sending an empty or repeated message; backing off 1h."}
        return {"action": "send", "body": text, "cta": result.get("cta") or "open_ended",
                "rationale": result.get("rationale") or "Continuing the conversation."}
    if result["action"] == "wait":
        return {"action": "wait", "wait_seconds": int(result.get("wait_seconds") or 3600),
                "rationale": result.get("rationale") or "Backing off."}
    return {"action": "end", "rationale": result.get("rationale") or "Closing conversation."}


@app.post("/v1/reply")
def reply(body: ReplyBody):
    deadline = time.monotonic() + REPLY_BUDGET_SECONDS
    with _lock:
        conv = conversations.get(body.conversation_id)
        if conv is None:
            # Unknown conversation (the simulator replies without a prior tick):
            # start a record from whatever the request tells us.
            conv = {"conversation_id": body.conversation_id, "merchant_id": body.merchant_id,
                    "customer_id": body.customer_id, "trigger_id": None,
                    "send_as": "merchant_on_behalf" if body.from_role == "customer" else "vera",
                    "status": "open", "messages": []}
            conversations[body.conversation_id] = conv
        if not conv.get("merchant_id") and body.merchant_id:
            conv["merchant_id"] = body.merchant_id
        conv["messages"].append({"from": body.from_role, "body": body.message,
                                 "ts": body.received_at or _now_iso()})
        merchant_id = conv.get("merchant_id")
        merchant_flags = {"opted_out": merchant_id in opted_out_merchants,
                          "auto_reply_count": auto_reply_counts.get(merchant_id, 0)}
        conv_snapshot = {**conv, "messages": list(conv["messages"])}

    merchant = _get("merchant", merchant_id)
    state = {
        "conversation": conv_snapshot,
        "merchant": merchant,
        "category": _get("category", (merchant or {}).get("category_slug")),
        "trigger": _get("trigger", conv_snapshot.get("trigger_id")),
        "customer": _get("customer", conv_snapshot.get("customer_id")),
        "merchant_flags": merchant_flags,
        "from_role": body.from_role,
        "turn_number": body.turn_number,
        "deadline": deadline,
    }

    try:
        result = conversation_handlers.respond(state, body.message)
    except Exception as e:
        log.exception("reply handler crashed: %s", e)
        result = None

    updates = (result or {}).pop("_state_updates", {}) if isinstance(result, dict) else {}
    response = _safe_reply(result, conv_snapshot)

    with _lock:
        # Apply side effects the handler asked for (keeps the handler itself pure).
        if merchant_id and updates.get("opt_out"):
            opted_out_merchants.add(merchant_id)
        if merchant_id and "auto_reply_count" in updates:
            auto_reply_counts[merchant_id] = int(updates["auto_reply_count"])
        if response["action"] == "send":
            conv["messages"].append({"from": "bot", "body": response["body"],
                                     "cta": response["cta"], "ts": _now_iso()})
        if response["action"] == "end":
            conv["status"] = "ended"

    return response


# ---------------------------------------------------------------------------
# POST /v1/teardown (optional) — wipe everything at the end of a test
# ---------------------------------------------------------------------------
@app.post("/v1/teardown")
async def teardown():
    global _conv_seq
    with _lock:
        contexts.clear()
        conversations.clear()
        sent_suppression_keys.clear()
        sent_trigger_ids.clear()
        opted_out_merchants.clear()
        auto_reply_counts.clear()
        _conv_seq = 0
    return {"status": "ok", "wiped": True}
