"""
conversation_handlers.py — decides the bot's next move when a merchant/customer replies.

respond(state, message) -> dict
  state   — built by bot.py: {"conversation", "merchant", "category", "trigger",
            "customer", "merchant_flags": {"opted_out", "auto_reply_count"},
            "from_role", "turn_number", "deadline"}
  returns — {"action": "send", "body", "cta", "rationale"}
          | {"action": "wait", "wait_seconds", "rationale"}
          | {"action": "end", "rationale"}
          plus optional "_state_updates" (e.g. {"opt_out": True}) that bot.py applies.
          Keeping side effects out of this module makes it easy to test.

Approach: classify with RULES first (fast, deterministic, can't be talked out of an
opt-out). Two separate flows, because a customer must never be pitched merchant work:

  MERCHANT                                   CUSTOMER (send_as merchant_on_behalf)
  1. opt-out / hostile -> end                1. stop -> end
  2. auto-reply -> note / wait 24h / end     2. picked a slot -> confirm warmly, close
  3. commitment -> SHOW the draft, CONFIRM   3. yes -> offer the slots / confirm
  4. question -> answer it (even if "busy")  4. question -> answer from data
  5. later -> wait                           5. later -> wait;  no -> end
  6. off-topic -> decline, back to the stage
  7. soft no -> end;  other -> grounded answer

The conversation STAGE (nothing offered yet / draft awaiting CONFIRM / done) is read
from what the bot already sent, so a detour (GST) resumes where it left off.
Every send ends with exactly one ask (unless it closes the conversation).
"""

import json
import re
import time

import composer
import facts as F
import llm
import templates

# ---------------------------------------------------------------------------
# Lexicons (lower-case; matched with word boundaries where it matters)
# ---------------------------------------------------------------------------
OPT_OUT = [
    r"\bstop\b", r"\bunsubscribe\b", r"not interested", r"don'?t (message|text|contact|send)",
    r"do not (message|text|contact|send)", r"\bspam\b", r"\buseless\b", r"leave me alone",
    r"\bbothering\b", r"\bharass", r"remove me", r"\bscam\b", r"\bfraud\b", r"band karo",
    r"mat bhejo", r"nahi chahiye", r"\bbakwas\b", r"shut up", r"\bidiot\b", r"\bstupid\b",
    r"\bnonsense\b", r"waste of (my )?time", r"\bf+u+c+k", r"\bdamn you\b", r"get lost",
]
AUTO_REPLY = [
    r"thank(s| you) for (contacting|reaching|your message|messaging)", r"will (respond|reply|get back)",
    r"get back to you", r"automated (assistant|message|reply|response)", r"\bauto[- ]?reply",
    r"out of (the )?office", r"currently (unavailable|away)", r"(our )?business hours",
    r"we have received your (message|query)", r"our team will", r"as soon as possible",
    r"jaankari ke liye", r"main ek automated", r"team tak pahuncha",
]
STRONG_INTENT = [
    r"let'?s do (it|this)", r"\blets do\b", r"\bgo ahead\b", r"\bproceed\b", r"\bsend it\b",
    r"\bset it up\b", r"\bsign me up\b", r"i want to join", r"\bdo it\b", r"\bbook it\b",
    r"\bconfirm(ed)?\b", r"\bplease (do|send|start|go)\b", r"\bstart (it|now)\b",
    r"kar do\b", r"karo\b", r"\bchalega\b", r"\bchalo\b", r"shuru kar", r"bhej do",
]
WEAK_INTENT = r"^\s*(yes|yeah|yep|yup|ya|ok|okay|okk+|sure|done|haan|haa|han|ji|theek hai|thik hai|perfect|great|good idea|sounds good|looks good)\b"
LATER = [
    (r"\bnext week\b", 14400), (r"\btomorrow\b|\bkal\b", 14400), (r"\bin a meeting\b|\bbusy\b", 7200),
    (r"\blater\b|\bbaad (me|mein)\b|\bnot now\b|\babhi nahi\b|\bthodi der\b", 3600),
]
OFF_TOPIC = [
    (r"\bgst\b", "GST filing"), (r"\bincome tax\b|\bitr\b|\btax(es)?\b", "tax filing"),
    (r"\bloan\b", "loans"), (r"\binsurance\b", "insurance"), (r"\baccount(ing|s)\b|\bbookkeeping\b", "accounting"),
    (r"\blawyer\b|\blegal\b|\bcourt\b", "legal matters"), (r"\bpassport\b|\bvisa\b", "that"),
    (r"\bpan card\b|\baadhaar\b", "that"), (r"\bcrypto\b|\bstock market\b|\bshares\b", "investments"),
]
SOFT_NO = [r"^\s*(no|nope|nahi|nah)\b", r"no thanks", r"not required", r"don'?t need", r"zaroorat nahi",
           r"mat karo", r"don'?t do"]
THANKS = r"^\s*(thanks|thank you|thx|ty|ok thanks|great thanks|shukriya|dhanyavaad|dhanyawad|👍|🙏|ok|okay|done|perfect)\W*$"

# Question / information request. A message like "I've been busy, what timings do you
# have?" is a question, not a postponement.
QUESTION_WORDS = (r"\b(what|when|which|where|how|why|who|kya|kab|kaun|kaunsa|kaise|kitna|kitne|kidhar|"
                  r"price|cost|charges?|fees?|rate|timings?|slots?|time|details?|available|availability|"
                  r"batao|bataiye|tell me)\b")
QUESTION_TOPICS = [
    ("price", r"\b(price|pricing|cost|costs|charges?|fees?|rate|kitna|kitne|how much|₹|rs\.?)\b"),
    ("timing", r"\b(timings?|time|slots?|when|kab|schedule|available|availability|day|hours?)\b"),
    ("count", r"\b(how many|kitne log|number of|count)\b"),
    ("effect", r"\b(work|kaam karega|results?|worth|help|benefit|sure|sach)\b"),
]

# Phrases the replay judge reads as "still qualifying" after a commitment. Note "do you"
# also matches inside "do your", so bodies must avoid that substring entirely.
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]
ACTION_WORDS = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]

# Words that only make sense to a merchant. A customer message must never contain them.
MERCHANT_ONLY_WORDS = re.compile(r"\b(google post|gbp|profile|campaign|peer|ctr|draft|vera|magicpin|"
                                 r"listing|analytics|your customers|your members|your patients)\b", re.I)

HINGLISH_MARKERS = (r"\b(hai|hain|nahi|nahin|karo|kar|kya|aap|aapka|aapki|aapke|mujhe|chahiye|haan|theek|thik|"
                    r"kal|baad|bhej|accha|acha|ji|kaise|kitna|hoga|mera|meri|hoon|yeh|ye|bahut|shukriya|"
                    r"dhanyavaad|liye|mein|sach|karega|karegi|raha|rahi|bataiye|batao|abhi|kab)\b")

# What "yes" means for each trigger kind (the deliverable we offered).
DELIVERABLE = {
    "research_digest": "the patient-education WhatsApp note",
    "regulation_change": "the compliance checklist for your team",
    "cde_opportunity": "the calendar reminder",
    "supply_alert": "the customer WhatsApp note about the recall",
    "perf_dip": "the offer + Google post",
    "perf_spike": "the follow-up Google post",
    "seasonal_perf_dip": "the member challenge message",
    "renewal_due": "your plan renewal",
    "festival_upcoming": "the festival Google post",
    "curious_ask_due": "the Google post",
    "winback_eligible": "the comeback offer post",
    "dormant_with_vera": "the profile update",
    "ipl_match_today": "tonight's match-night post",
    "review_theme_emerged": "the public review reply",
    "milestone_reached": "the thank-you WhatsApp asking regulars for reviews",
    "active_planning_intent": "the launch announcement",
    "category_seasonal": "the seasonal-essentials post",
    "gbp_unverified": "the verification steps",
    "competitor_opened": "the Google post built on your review strengths",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def respond(state: dict, message: str) -> dict:
    msg = (message or "").strip()
    low = msg.lower()
    conv = state.get("conversation") or {}
    is_customer = state.get("from_role") == "customer" or conv.get("send_as") == "merchant_on_behalf"
    ctx = _Ctx(state, msg, is_customer)
    out = _customer_flow(ctx, low) if is_customer else _merchant_flow(ctx, low)
    if out.get("action") == "send":
        out["body"] = _finalize_body(ctx, out["body"], out.get("cta"))
    return out


def _merchant_flow(ctx: "_Ctx", low: str) -> dict:
    flags = ctx.state.get("merchant_flags") or {}

    # 1. Opt-out / hostile — highest priority; never argue, never pitch.
    if _any(OPT_OUT, low):
        return {"action": "end",
                "rationale": "Merchant opted out / expressed frustration. Closing gracefully and "
                             "suppressing all proactive messages to this merchant.",
                "_state_updates": {"opt_out": True, "auto_reply_count": 0}}

    # 2. Auto-reply (canned WhatsApp Business text, or the same text repeated verbatim).
    if _is_auto_reply(low, ctx.conv):
        n = int(flags.get("auto_reply_count") or 0) + 1
        upd = {"auto_reply_count": n}
        if n == 1:
            body = ctx.say(
                f"Looks like an auto-reply 🙂 When the owner sees this, just reply YES and I'll get {ctx.deliverable} ready for {ctx.name}.",
                f"Lagta hai yeh auto-reply hai 🙂 Owner jab dekhein, bas YES reply karein — main {ctx.name} ke liye {ctx.deliverable} ready kar dungi.")
            return {**_send(ctx, body, "binary_yes_stop",
                            "Detected a WhatsApp Business auto-reply; one short prompt addressed to the owner, ending with a YES ask."),
                    "_state_updates": upd}
        if n == 2:
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Same auto-reply again: the owner isn't at the phone. Backing off 24h instead of burning turns.",
                    "_state_updates": upd}
        return {"action": "end", "rationale": f"Auto-reply {n} times in a row with no human response; closing the conversation.",
                "_state_updates": upd}

    updates = {"auto_reply_count": 0}  # a genuine human message resets the auto-reply streak
    question = _is_question(low)

    # 3. Commitment -> action mode (a question like "ok but how much?" is answered instead).
    if _is_commitment(low) and not question:
        if flags.get("opted_out"):
            updates["opt_out"] = False  # they came back and said yes: re-engage
        return {**_merchant_commitment(ctx), "_state_updates": updates}

    # 4. Off-topic -> decline in one line, then resume the current stage.
    topic = _off_topic(low)
    if topic:
        return {**_off_topic_reply(ctx, topic, opted_out=bool(flags.get("opted_out"))), "_state_updates": updates}

    # 5. A question is answered, even if the message also says "busy".
    if question:
        return {**_merchant_answer(ctx), "_state_updates": updates}

    # 6. Later / busy (actually postponing) -> back off.
    for pattern, seconds in LATER:
        if re.search(pattern, low):
            return {"action": "wait", "wait_seconds": seconds,
                    "rationale": f"Merchant postponed ('{ctx.msg[:40]}'); backing off {seconds // 3600}h.",
                    "_state_updates": updates}

    # 7. Soft no -> graceful exit (not an opt-out from everything).
    if _any(SOFT_NO, low):
        return {"action": "end", "rationale": "Merchant declined this suggestion; exiting gracefully without pushing.",
                "_state_updates": updates}

    # 8. Anything else -> grounded answer.
    return {**_merchant_answer(ctx), "_state_updates": updates}


def _customer_flow(ctx: "_Ctx", low: str) -> dict:
    # 1. Stop -> end. (Only this customer's thread; the merchant is not opted out.)
    if _any(OPT_OUT, low):
        return {"action": "end", "rationale": "Customer asked to stop; closing without further messages."}

    # 2. Already booked and they say thanks/ok -> close politely without another message.
    if ctx.stage == "booked" and re.search(THANKS, low):
        return {"action": "end", "rationale": "Booking already confirmed and the customer acknowledged; closing."}

    # 3. Picked a slot -> confirm warmly and close (no further ask, no upsell).
    slot = _picked_slot(ctx, low)
    if slot:
        body = ctx.say(f"Done — you're booked for {slot} at {ctx.name}. See you then! 🙂 If anything changes, just message us here.",
                       f"Done — {slot} ke liye aapki booking {ctx.name} mein confirm hai. Milte hain! 🙂 Kuch badle toh yahin message kar dijiye.")
        return _send(ctx, body, "none", f"Customer picked '{slot}'; confirmed the booking warmly and closed (no further ask).")

    question = _is_question(low)
    # 4. Yes / commitment without a specific slot.
    if _is_commitment(low) and not question:
        return _customer_commitment(ctx)

    # 5. Question (even with "busy") -> answer from the data.
    if question:
        return _customer_answer(ctx)

    for pattern, seconds in LATER:
        if re.search(pattern, low):
            return {"action": "wait", "wait_seconds": seconds,
                    "rationale": f"Customer postponed; backing off {seconds // 3600}h."}
    if _any(SOFT_NO, low):
        return {"action": "end", "rationale": "Customer declined; closing politely."}
    return _customer_answer(ctx)


# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------
class _Ctx:
    def __init__(self, state: dict, msg: str, is_customer: bool):
        self.state = state
        self.msg = msg
        self.is_customer = is_customer
        self.merchant = F.clean(state.get("merchant") or {})
        self.category = F.clean(state.get("category") or {})
        self.trigger = F.clean(state.get("trigger") or {})
        self.customer = F.clean(state.get("customer")) if state.get("customer") else None
        self.conv = state.get("conversation") or {}
        self.kind = self.trigger.get("kind")
        self.p = self.trigger.get("payload") or {}
        self.name = F.merchant_name(self.merchant) if self.merchant else "our team"
        self.sent = [m.get("body", "") for m in self.conv.get("messages", []) if m.get("from") == "bot"]
        self.deadline = state.get("deadline")
        self.lang = self._detect_lang(msg)
        self.offer, self.offer_source = (F.best_offer(self.merchant, self.category)
                                         if self.merchant else (None, None))
        self.deliverable = DELIVERABLE.get(self.kind) or (
            f"a Google post featuring your {self.offer}" if self.offer and self.offer_source == "merchant"
            else "the Google post")
        self.slots = [F.slot_label(s) for s in (self.p.get("available_slots") or self.p.get("next_session_options") or [])
                      if isinstance(s, dict)]
        self.slots = [s for s in self.slots if s]
        self.stage = self._stage()

    def _detect_lang(self, msg: str) -> str:
        """Per-turn language: Devanagari or Hinglish words -> hinglish; plain English -> en;
        very short/ambiguous replies ("ok") -> the default for this person."""
        if re.search(r"[ऀ-ॿ]", msg) or re.search(HINGLISH_MARKERS, msg.lower()):
            return "hinglish"
        if len(re.findall(r"[a-zA-Z]+", msg)) >= 3:
            return "en"
        if self.is_customer:
            return F.customer_lang(self.customer, self.merchant)
        return F.merchant_lang(self.category, self.merchant) if self.category else "en"

    def _stage(self) -> str:
        """Where the conversation is, judged from what the bot already sent."""
        low_sent = [s.lower() for s in self.sent]
        if any("you're booked" in s or "booking" in s and "confirm hai" in s for s in low_sent):
            return "booked"
        if any("going live" in s or "live ho raha" in s for s in low_sent):
            return "done"
        if any("reply confirm" in s or "confirm reply" in s or "confirm karein" in s for s in low_sent[-2:]):
            return "awaiting_confirm"
        return "pitched"

    def say(self, en: str, hinglish: str) -> str:
        return hinglish if self.lang == "hinglish" else en

    def topic(self) -> str:
        if self.kind:
            return F.humanize(self.kind).replace("perf ", "performance ")
        return f"{self.name}'s profile"


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def _any(patterns, low: str) -> bool:
    return any(re.search(p, low) for p in patterns)


def _is_auto_reply(low: str, conv: dict) -> bool:
    if _any(AUTO_REPLY, low):
        return True
    # Same inbound text repeated verbatim in this conversation = canned reply.
    inbound = [m.get("body", "").strip().lower() for m in conv.get("messages", []) if m.get("from") != "bot"]
    return len(inbound) >= 2 and len(low) > 20 and inbound[:-1].count(low) >= 1


NEGATION = r"\b(mat|nahi|nahin|don'?t|do not|not|never|no)\b"


def _is_commitment(low: str) -> bool:
    # "mat karo" / "don't do it" contain intent words but mean no.
    # ("no problem" / "koi baat nahi" are agreement idioms, not negations.)
    stripped = re.sub(r"no (problem|issues?|worries)|koi baat nahi|why not", "", low)
    if re.search(NEGATION, stripped) and not re.search(r"let'?s do (it|this)|\blets do\b", low):
        return False
    if _any(STRONG_INTENT, low):
        return True
    return bool(re.search(WEAK_INTENT, low))


def _is_question(low: str) -> bool:
    """A real question or information request. 'Ok let's do it. What's next?' is NOT a
    question to answer — 'what's next' is part of the commitment."""
    body = re.sub(r"what'?s next\??|whats next\??|aage kya\??", "", low)
    return "?" in body or bool(re.search(QUESTION_WORDS, body))


def _question_topic(low: str) -> str:
    for name, pattern in QUESTION_TOPICS:
        if re.search(pattern, low):
            return name
    return "other"


def _off_topic(low: str) -> str | None:
    for pattern, label in OFF_TOPIC:
        if re.search(pattern, low):
            return label
    return None


_ORDINALS = {1: r"\b(1|1st|first|pehla|pehli|one)\b", 2: r"\b(2|2nd|second|doosra|doosri|dusra|two)\b",
             3: r"\b(3|3rd|third|teesra|three)\b"}


def _picked_slot(ctx: "_Ctx", low: str) -> str | None:
    """Which offered slot the customer picked: '2', 'the first one', 'Thursday works',
    '5 Nov', '6pm'... None if no slots were offered or none matches."""
    if not ctx.slots or re.search(NEGATION, low):
        return None
    for i, pattern in _ORDINALS.items():
        if i <= len(ctx.slots) and re.search(pattern, low) and not re.search(r"\d{2,}|\d\s*(pm|am)", low):
            return ctx.slots[i - 1]
    for slot in ctx.slots:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z]{3,}|\d{1,2}(?::\d\d)?\s*(?:am|pm)|\d{1,2}\s+[A-Za-z]{3}", slot)]
        tokens = [t for t in tokens if t not in ("nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct")]
        if any(t.replace(" ", "") in low.replace(" ", "") for t in tokens):
            return slot
    return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _send(ctx: "_Ctx", body: str, cta: str, rationale: str) -> dict:
    return {"action": "send", "body": _unique(ctx, body), "cta": cta, "rationale": rationale}


def _unique(ctx: "_Ctx", body: str) -> str:
    """Anti-repetition: never send a body already sent in this conversation."""
    if body not in ctx.sent:
        return body
    for extra in (" (Quick follow-up.)", " — whenever you're ready.", " 🙏"):
        if body + extra not in ctx.sent:
            return body + extra
    return body + f" ({len(ctx.sent)})"


def _finalize_body(ctx: "_Ctx", body: str, cta: str | None) -> str:
    """Last pass on every outgoing body: clean spaces/signs, Vera's feminine Hindi,
    and exactly one clear ask at the end (unless this message closes the thread)."""
    body = composer.normalize(body)
    if not ctx.is_customer:
        body = F.feminine_vera(body)
    if cta and cta != "none":
        last = re.split(r"(?<=[.!?])\s+", body)[-1].lower()
        if not re.search(r"\?|\breply\b|\btell us\b|\blet us know\b|\bbataiye\b|\bbatayein\b|\bmessage us\b", last):
            body = body.rstrip() + " " + _ask_for_stage(ctx)
    return body


def _ask_for_stage(ctx: "_Ctx") -> str:
    """The one ask that fits where the conversation is."""
    if ctx.is_customer:
        if ctx.slots:
            return ctx.say("Reply 1 or 2 to pick a slot.", "Slot chunne ke liye 1 ya 2 reply karein.")
        return ctx.say("Reply YES and we'll book it.", "Book karne ke liye YES reply karein.")
    if ctx.stage == "awaiting_confirm":
        return ctx.say("Reply CONFIRM to make it live.", "Live karne ke liye CONFIRM reply karein.")
    return ctx.say("Reply YES to go ahead.", "Aage badhne ke liye YES reply karein.")


# ---------------------------------------------------------------------------
# Drafts: the ACTUAL short deliverable, built only from facts (shown on "yes")
# ---------------------------------------------------------------------------
def _draft(ctx: "_Ctx") -> str:
    """A short, ready-to-use draft for what was offered. Only facts from the data;
    anything unknown is left out rather than guessed."""
    m, cat, p = ctx.merchant, ctx.category, ctx.p
    name, loc = ctx.name, F.locality(m)
    where = f"{name}, {loc}" if loc else name
    offers = F.active_offers(m)
    kind = ctx.kind
    item = F.digest_for_trigger(cat, ctx.trigger) if ctx.trigger else None

    if kind == "research_digest" and item:
        return (f"New research ({item.get('source')}): {item.get('title')}. "
                f"Ask us about the right check-up interval for you at your next visit. — {name}")
    if kind == "regulation_change" and item:
        dl = F.fmt_date(p.get("deadline_iso") or item.get("date"), with_year=True)
        return (f"Checklist — {item.get('title')}:\n1. {(item.get('actionable') or 'Review the change').rstrip('.')}\n"
                f"2. Brief the team{' before ' + dl if dl else ''}\n3. File the updated SOP")
    if kind == "cde_opportunity" and item:
        return f"Reminder: {item.get('title')} — {F.fmt_slot(item.get('date')) or ''}. {(item.get('actionable') or '').rstrip('.')}.".replace(" — .", ".")
    if kind == "supply_alert":
        mol, batches = p.get("molecule") or "your medicine", ", ".join(p.get("affected_batches") or [])
        return (f"Namaste! {name} here. A voluntary recall has been issued for {mol}"
                f"{' batches ' + batches if batches else ''}. If you bought {mol} from us recently, please bring "
                f"your pack in and we'll check the batch and arrange a replacement.")
    if kind == "review_theme_emerged":
        theme = F.humanize(p.get("theme") or "your experience")
        return (f"Thank you for the feedback. We're sorry about the {theme} — we're looking into it "
                f"and would love the chance to serve you better. — {name}")
    if kind == "milestone_reached":
        return (f"Thank you for being a regular at {name}! If you enjoyed your last visit, a quick "
                f"Google review would mean a lot to us. 🙏")
    if kind == "ipl_match_today" and p.get("match"):
        return f"Watching {p['match']} at home tonight? Order from {where} for delivery. 🏏"
    if kind == "category_seasonal":
        items = [F.humanize(re.sub(r"_demand_[+-]\d+$", "", t)) for t in p.get("trends") or [] if "+" in str(t)]
        extra = next((o for o in offers if "delivery" in o.lower()), None)
        return (f"Summer essentials in stock at {where}: {', '.join(items) or 'seasonal essentials'}."
                + (f" {extra}." if extra else ""))
    if kind == "active_planning_intent":
        tctx = templates.Ctx(cat, m, ctx.trigger, None, None)
        proposal, _ = templates.history_facts(tctx)
        topic = F.humanize(p.get("intent_topic")) or "our new program"
        return (f"New at {where}: {topic}" + (f" — {proposal}" if proposal else "") + "."
                + (f" {offers[0]}." if offers else "") + " Message us to book.")
    if kind == "seasonal_perf_dip":
        return (f"{name} Stay-on-Track Challenge: keep your usual sessions going this month and check in with "
                f"us after each one — we'll cheer every check-in. Reply JOIN to take part!")
    if kind == "gbp_unverified":
        path = F.humanize(p.get("verification_path")) or "postcard or phone call"
        return (f"Steps: 1. Open your Google Business Profile 2. Choose verification by {path} "
                f"3. Enter the code you receive — done.")
    if kind == "renewal_due":
        sub = m.get("subscription") or {}
        amount = F.fmt_money(p.get("renewal_amount"))
        return f"Renewal: {p.get('plan') or sub.get('plan') or 'current'} plan" + (f", {amount}" if amount else "") + "."
    # Default: a short Google post around the live offer (or the business itself).
    offer = offers[0] if offers else None
    return f"{where}" + (f": {offer}." if offer else " — we'd love to see you this week.") + " Call or WhatsApp us to book."


# ---------------------------------------------------------------------------
# Merchant replies
# ---------------------------------------------------------------------------
def _merchant_commitment(ctx: "_Ctx") -> dict:
    """The merchant said yes. If a draft is awaiting CONFIRM, execute; otherwise SHOW the
    actual draft now and ask for CONFIRM. Never a qualifying question."""
    if ctx.stage == "awaiting_confirm":
        body = ctx.say(
            f"Confirmed — {ctx.deliverable} for {ctx.name} is going live now. I'll share how it's doing here next week.",
            f"Confirmed! {ctx.name} ke liye {ctx.deliverable} ab live ho raha hai. Next week yahin update bhejungi.")
        return _send(ctx, body, "none", "Merchant confirmed the draft; executing and closing the loop (no further ask needed).")

    llm_out = _llm_reply(ctx, mode="commitment")
    if llm_out:
        return llm_out
    draft = _draft(ctx)
    body = ctx.say(f"Done — here's the draft for {ctx.deliverable}:\n\n\"{draft}\"\n\nReply CONFIRM to make it live.",
                   f"Done — {ctx.deliverable} ka draft yeh raha:\n\n\"{draft}\"\n\nLive karne ke liye CONFIRM reply karein.")
    return _send(ctx, body, "binary_confirm_cancel",
                 "Merchant committed; switched to action mode and showed the actual draft (built from facts) with a single CONFIRM ask.")


def _off_topic_reply(ctx: "_Ctx", topic: str, opted_out: bool) -> dict:
    thing = topic if topic != "that" else "That"
    cap = thing[0].upper() + thing[1:]
    if opted_out:
        body = ctx.say(f"{cap} is best handled by your CA or a specialist — it's outside what I can help with. I won't send further updates unless you message me.",
                       f"{thing} ke liye aapke CA ya specialist best rahenge — yeh mere scope se bahar hai. Aap message karein tabhi aage update bhejungi.")
        return _send(ctx, body, "none", "Merchant had opted out; answered the out-of-scope question politely without pitching.")
    if ctx.stage == "awaiting_confirm":
        # They already said yes: resume the committed state instead of re-asking for YES.
        body = ctx.say(f"{cap} is best handled by your CA — that's outside what I can help with. Back to {ctx.deliverable}: the draft above is ready. Reply CONFIRM to make it live.",
                       f"{thing} ke liye aapke CA best rahenge — yeh mere scope se bahar hai. Wapas {ctx.deliverable} par: upar wala draft ready hai. Live karne ke liye CONFIRM reply karein.")
        return _send(ctx, body, "binary_confirm_cancel", f"Out-of-scope ask ({topic}) declined; resumed the pending draft (merchant had already committed).")
    body = ctx.say(f"{cap} is best handled by your CA — that's outside what I can help with. Back to the {ctx.topic()}: shall I go ahead with {ctx.deliverable}? Reply YES.",
                   f"{thing} ke liye aapke CA best rahenge — yeh mere scope se bahar hai. Wapas {ctx.topic()} par: {ctx.deliverable} ke saath aage badhun? Reply YES.")
    return _send(ctx, body, "binary_yes_stop", f"Out-of-scope ask ({topic}) declined in one line; steered back to the original trigger.")


def _merchant_answer(ctx: "_Ctx") -> dict:
    """Answer the merchant's question (LLM first, topic-aware template fallback)."""
    llm_out = _llm_reply(ctx, mode="engaged")
    if llm_out:
        return llm_out
    topic = _question_topic(ctx.msg.lower())
    ask = _ask_for_stage(ctx)
    m, cat = ctx.merchant, ctx.category
    tctx = templates.Ctx(cat, m, ctx.trigger or {}, None, None) if m and cat else None

    if topic == "price":
        prices = _known_prices(ctx)
        if prices:
            body = ctx.say(f"Keep it anchored on what customers already know: {prices}. "
                           f"For anything new, set the final price yourself — I'll use exactly what you approve. {ask}",
                           f"Jo customers pehle se jaante hain usi par anchor rakhiye: {prices}. "
                           f"Naye item ka final price aap tay kijiye — main wahi use karungi jo aap approve karein. {ask}")
            why = f"Price question answered with real prices from the data ({prices}); no new price invented."
        else:
            body = ctx.say(f"There's no price on file for this yet, so I won't guess one — tell me the price you want and I'll put it in the draft. {ask}",
                           f"Iska koi price abhi record mein nahi hai, isliye main guess nahi karungi — aap price bataiye, main draft mein daal dungi. {ask}")
            why = "Price question: no price in the data, said so honestly instead of inventing one."
        return _send(ctx, body, "binary_confirm_cancel" if ctx.stage == "awaiting_confirm" else "binary_yes_stop", why)

    if topic == "timing":
        when = _known_timing(ctx)
        body = ctx.say((f"{when}. " if when else "It can go out as soon as you approve it — today if you like. ") + ask,
                       (f"{when}. " if when else "Aap approve karein, aaj hi live ho sakta hai. ") + ask)
        return _send(ctx, body, "binary_yes_stop", "Timing question answered from the dates in the data.")

    if topic == "count":
        counts = _known_counts(ctx)
        if counts:
            body = ctx.say(f"From your data: {counts}. {ask}", f"Aapke data ke hisaab se: {counts}. {ask}")
            return _send(ctx, body, "binary_yes_stop", f"Count question answered from customer_aggregate ({counts}).")

    anchor = templates.perf_anchor(tctx, positive=(topic == "effect")) if tctx else ""
    body = ctx.say((f"Fair question. What the data shows: {anchor}. " if anchor else "Fair question. ") + ask,
                   (f"Sahi sawaal. Data yeh kehta hai: {anchor}. " if anchor else "Sahi sawaal. ") + ask)
    return _send(ctx, body, "binary_yes_stop", "Engaged reply (template): answered with a real number from the data and one next step.")


def _known_prices(ctx: "_Ctx") -> str:
    """Real prices only: live offers, prices Vera already proposed in chat, digest prices."""
    out = [o for o in F.active_offers(ctx.merchant) if "₹" in o]
    tctx = templates.Ctx(ctx.category, ctx.merchant, ctx.trigger or {}, None, None) if ctx.merchant else None
    if tctx:
        proposal, _ = templates.history_facts(tctx)
        for price in re.findall(r"₹\s?[\d,]+", proposal or ""):
            out.append(f"{price} (the price we discussed earlier)")
    return "; ".join(dict.fromkeys(out))[:220]


def _known_timing(ctx: "_Ctx") -> str:
    p = ctx.p
    if ctx.slots:
        return "Open slots: " + " or ".join(ctx.slots)
    for key in ("deadline_iso", "match_time_iso", "date", "due_date", "wedding_date"):
        if p.get(key):
            when = F.fmt_slot(p[key]) if "T" in str(p[key]) else F.fmt_date(p[key], with_year=True)
            return f"The key date here is {when}"
    return ""


def _known_counts(ctx: "_Ctx") -> str:
    agg = ctx.merchant.get("customer_aggregate") or {}
    labels = {"high_risk_adult_count": "high-risk adult patients", "total_active_members": "active members",
              "chronic_rx_count": "chronic-Rx customers", "total_unique_ytd": "unique customers this year",
              "lapsed_180d_plus": "customers lapsed 180+ days", "lapsed_90d_plus": "customers lapsed 90+ days"}
    return ", ".join(f"{F.fmt_int(agg[k])} {v}" for k, v in labels.items() if isinstance(agg.get(k), (int, float)))[:200]


# ---------------------------------------------------------------------------
# Customer replies (always as the business; never merchant deliverables)
# ---------------------------------------------------------------------------
def _customer_commitment(ctx: "_Ctx") -> dict:
    if ctx.kind == "chronic_refill_due":
        body = ctx.say(f"Done — your order is confirmed and will be dispatched from {ctx.name}. We'll message you when it's out for delivery.",
                       f"Done — aapka order confirm ho gaya, {ctx.name} se dispatch hoga. Delivery ke liye nikalte hi message karenge.")
        return _send(ctx, body, "none", "Customer confirmed the refill; confirmed dispatch as the pharmacy and closed.")
    if ctx.slots:
        options = " or ".join(f"{i}) {s}" for i, s in enumerate(ctx.slots, 1))
        body = ctx.say(f"Great! Which works better: {options}? Reply 1 or 2.",
                       f"Badhiya! Kaunsa time theek rahega: {options}? 1 ya 2 reply karein.")
        return _send(ctx, body, "multi_choice_slot", "Customer agreed; asked them to pick one of the given slots.")
    pref = F.preferred_slot(ctx.customer)
    body = ctx.say(f"Great — we'll hold a {pref + ' ' if pref else ''}slot for you this week. The {ctx.name} team will message the exact time here. 🙂",
                   f"Badhiya — is hafte aapke liye {pref + ' ka ' if pref else ''}slot hold kar rahe hain. {ctx.name} team yahin exact time bhej degi. 🙂")
    return _send(ctx, body, "none", "Customer agreed; confirmed a held slot (preferred time where known) and closed without inventing a time.")


def _customer_answer(ctx: "_Ctx") -> dict:
    topic = _question_topic(ctx.msg.lower())
    if topic == "timing":
        if ctx.slots:
            options = " or ".join(f"{i}) {s}" for i, s in enumerate(ctx.slots, 1))
            body = ctx.say(f"We have {options}. Reply 1 or 2 and it's yours.",
                           f"Hamare paas {options} hai. 1 ya 2 reply karein, slot aapka.")
            return _send(ctx, body, "multi_choice_slot", "Customer asked about timings; offered the actual open slots.")
        pref = F.preferred_slot(ctx.customer)
        body = ctx.say(f"We can fit you in {('on a ' + pref) if pref else 'this week'} — tell us the day that suits you and we'll book it.",
                       f"Hum aapko {('' + pref + ' mein') if pref else 'is hafte'} fit kar sakte hain — jo din suit kare woh bataiye, hum book kar denge.")
        return _send(ctx, body, "open_ended", "Customer asked about timings; offered their preferred time window without inventing exact times.")
    if topic == "price":
        offers = F.active_offers(ctx.merchant)
        if offers:
            body = ctx.say(f"Right now at {ctx.name}: {'; '.join(offers[:2])}. Reply YES and we'll book you in.",
                           f"Abhi {ctx.name} mein: {'; '.join(offers[:2])}. Book karne ke liye YES reply karein.")
            return _send(ctx, body, "binary_yes_stop", "Customer asked about price; quoted the business's live offers only.")
    llm_out = _llm_reply(ctx, mode="engaged")
    if llm_out:
        return llm_out
    body = ctx.say(f"Thanks for asking! The {ctx.name} team will reply to this here shortly. Reply YES if you'd like us to book you in meanwhile.",
                   f"Poochne ke liye shukriya! {ctx.name} team jaldi hi yahin jawab degi. Tab tak booking karni ho toh YES reply karein.")
    return _send(ctx, body, "binary_yes_stop", "Customer question not answerable from the data; routed to the business team, one booking ask.")


# ---------------------------------------------------------------------------
# LLM replies (grounded + validated; None -> caller uses its template)
# ---------------------------------------------------------------------------
REPLY_SYSTEM = """You are Vera, magicpin's WhatsApp assistant for Indian merchants, replying mid-conversation.
Output a JSON object: {"body": "...", "cta": "...", "rationale": "..."}.
Rules:
- Use ONLY facts from FACT_SHEET and the conversation. Never invent numbers, prices, dates, names, days of the week, "free", causes, or safety claims.
- Answer the latest message directly first (if they asked about price, quote the real prices from the facts; if timings, the real dates/slots), then give exactly ONE next step as the last sentence.
- Exactly one ask, in one language. Don't repeat the ask in Hindi and English.
- Don't re-introduce yourself, no greeting preamble, no URLs, don't repeat an earlier message.
- Only offer what Vera can do: draft messages/posts/checklists/replies, set reminders, share summaries, set up an offer or post. Never claim to filter customer lists, access records, call people or take payments.
- Vera is female: Hindi first-person forms are feminine (karungi, bhejungi, kar rahi hoon).
- Never output placeholders like <...>, {...} or [...].
- Max 400 characters. language: "hinglish" = natural Hindi-English in Roman script; "en" = English.
cta must be one of: binary_yes_stop, binary_confirm_cancel, open_ended, multi_choice_slot, none."""

COMMITMENT_RULES = """The merchant has just COMMITTED ("yes / let's do it"). Switch to ACTION mode:
- Start with "Done" or "Here's".
- SHOW the actual short draft of the deliverable in quotes (2-3 lines, built only from FACT_SHEET facts) — don't just say you're preparing it.
- Ask NO question. Never use the phrases: "would you", "do you", "do your", "can you tell", "what if", "how about".
- Last sentence exactly: "Reply CONFIRM to make it live." cta = binary_confirm_cancel."""

CUSTOMER_RULES = """You are writing to a CUSTOMER as the business (e.g. "Dr. Meera's Dental Clinic"), never as Vera or magicpin.
Never mention posts, drafts, profiles, campaigns, peers or any merchant analytics. Help with booking, timings, prices from their live offers.
If they picked a time, confirm it warmly and close (cta none)."""


def _llm_reply(ctx: "_Ctx", mode: str) -> dict | None:
    if not (llm.enabled() and ctx.merchant and ctx.category):
        return None
    budget = llm.default_timeout()
    if ctx.deadline:
        budget = min(budget, ctx.deadline - time.monotonic() - 0.3)
    if budget < 1.5:
        return None
    trigger = ctx.trigger or {"kind": "conversation", "scope": "customer" if ctx.is_customer else "merchant", "payload": {}}
    if ctx.is_customer:
        trigger = {**trigger, "scope": "customer"}  # forces the customer-safe fact sheet
    sheet = composer.build_fact_sheet(ctx.category, ctx.merchant, trigger, ctx.customer, None)
    history = "\n".join(f"{m.get('from')}: {m.get('body')}" for m in ctx.conv.get("messages", [])[-6:])
    if ctx.is_customer:
        instructions = CUSTOMER_RULES
    elif mode == "commitment":
        instructions = COMMITMENT_RULES + f"\nDeliverable to draft: {ctx.deliverable}.\nA verified draft you may polish (keep its facts): \"{_draft(ctx)}\""
    else:
        instructions = (f"Conversation stage: {ctx.stage}. Deliverable on the table: {ctx.deliverable}.\n"
                        "Reply helpfully to the merchant's latest message, grounded in the facts, and propose one next step"
                        + (" (the pending draft: 'Reply CONFIRM to make it live')." if ctx.stage == "awaiting_confirm" else "."))
    user = (f"FACT_SHEET (JSON): {json.dumps(sheet, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"CONVERSATION SO FAR:\n{history}\n\nlanguage: {ctx.lang}\n{instructions}\nReturn the JSON object now.")
    result = llm.complete_json(REPLY_SYSTEM, user, timeout=budget, max_tokens=600)
    if not result:
        return None
    body = composer.normalize(str(result.get("body") or ""))
    if not ctx.is_customer:
        body = F.feminine_vera(body)
    facts_text = json.dumps(sheet, ensure_ascii=False) + " " + ctx.deliverable + " " + _draft(ctx) + " " + \
        " ".join(m.get("body", "") for m in ctx.conv.get("messages", []))
    if _validate_reply(ctx, body, facts_text, mode):
        return None
    cta = result.get("cta") if result.get("cta") in composer.ALLOWED_CTAS else "open_ended"
    if mode == "commitment":
        cta = "binary_confirm_cancel"
        if "confirm" not in body.lower()[-80:]:
            body = body.rstrip(" .") + ". Reply CONFIRM to make it live."
    return {"action": "send", "body": body, "cta": cta,
            "rationale": (str(result.get("rationale") or "").strip() + " [LLM reply, validated against facts.]").strip()}


def _validate_reply(ctx: "_Ctx", body: str, facts_text: str, mode: str) -> list[str]:
    low = body.lower()
    problems = []
    if not body or len(body) > 520:
        problems.append("empty or too long")
    if re.search(r"https?://|www\.", low):
        problems.append("url")
    if body in ctx.sent:
        problems.append("repeat")
    for w in F.taboo_words(ctx.category):
        core = re.sub(r"\s*\(.*?\)\s*", "", w).strip().lower()
        if core and core in low:
            problems.append("taboo")
    known = composer._numbers(facts_text)
    # Commitment replies deliver something concrete, and drafted lists are specs:
    # every number there must be known (no free pass for small numbers).
    strict = mode == "commitment" or composer.is_drafted_plan(body, None)
    invented = [n for n in composer._numbers(body) if n not in known
                and (strict or not (float(n).is_integer() and float(n) <= 10))]
    invented += [n for n in composer._unit_numbers(body) if n not in known and n not in invented]
    if invented:
        problems.append(f"invented numbers {invented[:3]}")
    problems += composer.common_text_problems(body, facts_text)
    if ctx.is_customer and MERCHANT_ONLY_WORDS.search(body):
        problems.append("merchant-only wording in a customer message")
    if mode == "commitment":
        if any(q in low for q in QUALIFYING) or "?" in body:
            problems.append("qualifying question after commitment")
        if not any(a in low for a in ACTION_WORDS):
            problems.append("no action word")
        if '"' not in body and "“" not in body:
            problems.append("no draft shown")
    return problems
