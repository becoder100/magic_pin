"""
composer.py — turns (category, merchant, trigger, customer) into one message.

Pipeline ("rules decide, AI writes"):
  1. Template draft   templates.render()  — deterministic, always valid.
  2. Fact sheet       build_fact_sheet()  — ONLY verified facts, pre-formatted
                                            ("38%", "2,100", "₹299"), trimmed to
                                            what this trigger needs (small prompt).
  3. LLM              llm.complete_json() — rewrites the draft into a sharper
                                            message using only the fact sheet.
  4. Validator        validate()          — rejects any number not in the facts,
                                            taboo words, URLs, empty/over-long
                                            bodies, a missing "Dr." for dentists...
  5. Fallback         if the LLM is off, slow, rate-limited or fails validation,
                      the template draft from step 1 is sent instead.

compose() must be deterministic for the same inputs (temperature 0 + cache)
and must never raise.
"""

import hashlib
import json
import re
import threading
import time

import facts as F
import llm
import templates

ALLOWED_CTAS = {"binary_yes_stop", "binary_confirm_cancel", "open_ended", "multi_choice_slot", "none"}
MAX_BODY_CHARS = 650

_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def compose(category: dict, merchant: dict, trigger: dict,
            customer: dict | None = None, now: str | None = None,
            deadline: float | None = None) -> dict:
    """Returns: body, cta, send_as, suppression_key, rationale, template_name, template_params.

    `deadline` is a time.monotonic() value: the LLM call must finish before it.
    """
    category, merchant = F.clean(category or {}), F.clean(merchant or {})
    trigger, customer = F.clean(trigger or {}), F.clean(customer) if customer else None

    key = _cache_key(category, merchant, trigger, customer, now)
    with _cache_lock:
        if key in _cache:
            return dict(_cache[key])

    draft = templates.render(category, merchant, trigger, customer, now)
    if not llm.enabled():
        return draft  # template-only mode (don't cache: a key may be added later)

    budget = llm.default_timeout()
    if deadline is not None:
        budget = min(budget, deadline - time.monotonic() - 0.3)
    if budget < 1.5:
        draft["rationale"] += " (Template used: not enough time left for the LLM.)"
        return draft

    sheet = build_fact_sheet(category, merchant, trigger, customer, now)
    result = llm.complete_json(SYSTEM_PROMPT, build_user_prompt(sheet, draft), timeout=budget)
    if not result:
        draft["rationale"] += f" (Template used: LLM unavailable — {llm.last_status}.)"
        return draft

    body = normalize(str(result.get("body") or ""))
    if trigger.get("scope") != "customer":
        body = F.feminine_vera(body)  # Vera's Hindi first person is always feminine
    problems = validate(body, sheet, draft, category, trigger)
    if problems:
        draft["rationale"] += f" (Template used: LLM draft rejected — {'; '.join(problems[:3])}.)"
        return draft

    cta = result.get("cta") if result.get("cta") in ALLOWED_CTAS else draft["cta"]
    body = _ensure_reply_instruction(body, cta)
    out = {**draft, "body": body, "cta": cta,
           "rationale": _rationale(result.get("rationale"), draft)}
    with _cache_lock:
        _cache[key] = dict(out)  # only cache LLM successes; failures may succeed next time
    return out


def fallback_compose(category: dict, merchant: dict, trigger: dict,
                     customer: dict | None = None, now: str | None = None) -> dict:
    """Deterministic, fast, never fails. Used when compose() runs out of time."""
    return templates.render(category, merchant, trigger, customer, now)


def _ensure_reply_instruction(body: str, cta: str) -> str:
    """Binary CTAs must tell the reader exactly what to reply (the judge rewards a
    single clear binary ask in the last sentence). Add it if the LLM left it out."""
    word = {"binary_yes_stop": "YES", "binary_confirm_cancel": "CONFIRM"}.get(cta)
    # Look at the whole final sentence (not just the last few characters) so we never
    # append a second "Reply YES" after "Reply YES and I'll turn this into ...".
    last_sentence = re.split(r"(?<=[.!?])\s+", body.strip())[-1] if body.strip() else ""
    if word and "reply" not in last_sentence.lower() and "reply" not in body.lower()[-200:]:
        body = body.rstrip()
        if body and body[-1] not in ".?!":
            body += "."
        body += f" Reply {word}."
    return body


def _cache_key(*parts) -> str:
    # `now` is reduced to the date: the same trigger on the same day gets the same message.
    *ctx, now = parts
    blob = json.dumps([ctx, (now or "")[:10]], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rationale(llm_rationale, draft: dict) -> str:
    base = str(llm_rationale or "").strip()
    return (base + " " if base else "") + f"[LLM-written from verified fact sheet; validated. {draft['rationale']}]"


# ---------------------------------------------------------------------------
# Fact sheet: only verified, pre-formatted facts relevant to this trigger
# ---------------------------------------------------------------------------
PCT_KEYS = re.compile(r"(_pct$|^ctr$|_ctr$|delta_yoy$|uplift)", re.I)


DIRECTIONAL_KEYS = re.compile(r"(delta|change)", re.I)


def fmt_value(key: str, v, directional: bool = False):
    """Pre-format numbers so the LLM copies them instead of converting (and erring).
    Changes are written as words, never signs: -0.3 -> "down 30%", 0.18 -> "up 18%".
    (Signed values made the model write "fell -30%".)"""
    directional = directional or bool(DIRECTIONAL_KEYS.search(key))
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        if PCT_KEYS.search(key) and abs(v) <= 5:
            if directional:
                return f"{'down' if v < 0 else 'up'} {F.fmt_pct(abs(v))}"
            return F.fmt_pct(v)
        return F.fmt_int(v) if float(v).is_integer() or abs(v) >= 100 else v
    if isinstance(v, dict):
        return {k: fmt_value(k, x, directional) for k, x in v.items()}
    if isinstance(v, list):
        return [fmt_value(key, x, directional) for x in v]
    return v


# Ambiguous payload keys renamed so the model can't misread them
# (e.g. it read vs_baseline: 18 as "18 calls total").
PAYLOAD_LABELS = {
    "vs_baseline": "usual_level_before_this_change",
    "delta_pct": "change_pct",
}


def build_fact_sheet(category, merchant, trigger, customer, now) -> dict:
    kind = trigger.get("kind")
    customer_facing = trigger.get("scope") == "customer"
    ident = merchant.get("identity") or {}
    agg = merchant.get("customer_aggregate") or {}
    sheet = {
        "today": (now or "")[:10] or None,
        "audience": "customer (you write AS the merchant's business, never as Vera)" if customer_facing
                    else "merchant owner (you are Vera, magicpin's assistant)",
        "trigger": {
            "kind": kind,
            "urgency": trigger.get("urgency"),
            "placeholder_payload": F.is_placeholder(trigger),
            "payload": fmt_value("payload", {PAYLOAD_LABELS.get(k, k): v for k, v in (trigger.get("payload") or {}).items()
                                             if k not in ("placeholder", "metric_or_topic")}),
        },
        "merchant": {
            "business_name": ident.get("name"),
            "salutation": F.salutation(category, merchant),
            "owner_first_name": F.owner_first_name(merchant),
            "locality": ident.get("locality"), "city": ident.get("city"),
            "language_mode": "english_with_light_hindi" if F.merchant_lang(category, merchant) == "hinglish" else "english",
            "subscription": fmt_value("subscription", merchant.get("subscription") or {}),
            "performance_30d": fmt_value("performance", F.perf(merchant)),
            "peer_comparison": [
                f"{g['metric']}: {F.fmt_metric(g['metric'], g['mine'])} vs peer avg {F.fmt_metric(g['metric'], g['peer'])}"
                + (" (below peer)" if g["below"] else " (at/above peer)")
                for g in F.peer_gaps(merchant, category)] + [
                f"{a['label']}: {a['mine']} vs peer avg {a['peer']} ({'better' if a['better'] else 'worse'} than peers)"
                for a in F.aggregate_vs_peer(merchant, category)],
            "active_offers": F.active_offers(merchant),
            "customer_aggregate": fmt_value("aggregate", agg),
            "signals": merchant.get("signals") or [],
            "review_themes": merchant.get("review_themes") or [],
            "last_conversation_turns": [f"{h.get('from')}: {h.get('body')}"
                                        for h in (merchant.get("conversation_history") or [])[-2:]],
        },
        "category": {
            "slug": category.get("slug"),
            "voice_tone": (category.get("voice") or {}).get("tone"),
            "voice_register": (category.get("voice") or {}).get("register"),
            "taboo_words": F.taboo_words(category),
            "catalog_offers_suggestions_only": [o.get("title") for o in category.get("offer_catalog") or []][:6],
        },
    }
    tp = sheet["trigger"]["payload"]
    raw_payload = trigger.get("payload") or {}
    # Slot labels with a weekday that contradicts the date lose the weekday.
    for key in ("available_slots", "next_session_options"):
        if isinstance(raw_payload.get(key), list):
            tp[key] = [F.slot_label(s) for s in raw_payload[key] if isinstance(s, dict)]
    # IPL: say "weekend", never guess the weekday (26 Apr 2026 is a Sunday, not "Saturday").
    if "is_weeknight" in raw_payload:
        tp["day_type"] = "weeknight" if raw_payload["is_weeknight"] else "weekend (weekday name not given)"
    # For a metric change, give the CURRENT value too, so a baseline or peer number is
    # never presented as the merchant's own count ("dropped 50% (from 12)").
    metric = raw_payload.get("metric")
    if metric and metric in F.perf(merchant):
        tp["current_value_last_30d"] = F.fmt_metric(metric, F.perf(merchant)[metric])
    item = F.digest_for_trigger(category, trigger)
    if item:
        sheet["digest_item"] = fmt_value("digest", item)
    beat = F.seasonal_beat(category, F.now_month(now))
    if beat:
        sheet["category"]["seasonal_beat_now"] = beat
    if kind in ("category_seasonal", "festival_upcoming", "research_digest", "curious_ask_due") or F.is_placeholder(trigger):
        sheet["category"]["trend_signals"] = fmt_value("trend", (category.get("trend_signals") or [])[:3])
    if kind == "ipl_match_today":
        ipl = next((d for d in category.get("digest") or [] if "ipl" in (d.get("title") or "").lower()), None)
        if ipl:
            sheet["digest_item"] = ipl
    # Computed dates (only when the judge gave us `now`).
    days = {}
    for k, v in (trigger.get("payload") or {}).items():
        if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v):
            d = F.days_until(v, now)
            if d is not None and d >= 0:
                days[k] = f"{d} days from today"
    # The payload may carry its own day counts (days_until, days_to_wedding...) computed
    # when the trigger was created. If they disagree with ours, give the model NEITHER
    # (only absolute dates): the judge checks counts against the payload.
    given = {k: v for k, v in (trigger.get("payload") or {}).items()
             if k.startswith("days_") and isinstance(v, (int, float))}
    ours = {int(v.split()[0]) for v in days.values()}
    if given and days and not any(abs(g - o) <= 1 for g in given.values() for o in ours):
        days = {}
        sheet["trigger"]["payload"] = {k: v for k, v in sheet["trigger"]["payload"].items() if k not in given}
    if days:
        sheet["trigger"]["computed_days_until"] = days
    if customer_facing:
        # A customer must never see the merchant's internal data (magicpin plan,
        # CTR, peer benchmarks, signals, Vera chat history). Keep only what the
        # business itself would say to a customer.
        safe = ("business_name", "owner_first_name", "locality", "city", "active_offers")
        sheet["merchant"] = {k: v for k, v in sheet["merchant"].items() if k in safe}
        # Category industry data (seasonal beats, trends, catalog) is for merchants,
        # not customers ("+50% pediatric appointments" means nothing to a patient).
        sheet["category"] = {k: v for k, v in sheet["category"].items()
                             if k in ("slug", "voice_tone", "voice_register", "taboo_words")}
        if sheet.get("digest_item") and kind != "chronic_refill_due":
            sheet.pop("digest_item")
        addr, subj = F.customer_names(customer)
        rel = (customer or {}).get("relationship") or {}
        sheet["customer"] = None if not customer else {
            "address_as": addr, "about": subj,
            "language_mode": "hinglish" if F.customer_lang(customer, merchant) == "hinglish" else "english",
            "state": customer.get("state"),
            "last_visit": F.fmt_date(rel.get("last_visit"), with_year=True),
            # Pre-computed so the model never does date arithmetic itself.
            "time_since_last_visit": _gap_text(F.days_until(now, rel.get("last_visit")) if now else None),
            "visits_total": rel.get("visits_total"),
            "services_received": [s for s in rel.get("services_received") or [] if s != "..."],
            "preferences": customer.get("preferences") or {},
            "senior_citizen": (customer.get("identity") or {}).get("senior_citizen"),
        }
        natural = templates.NATURAL_FIT.get(kind)
        if natural and category.get("slug") not in natural:
            sheet["trigger"]["note"] = (f"'{kind}' is unusual for a {category.get('slug')} business: "
                                        "use neutral wording like 'your regular follow-up is due'.")
    return _prune(sheet)


def _gap_text(days) -> str | None:
    """179 -> 'about 5 months (179 days)'; 20 -> 'about 2 weeks (20 days)'."""
    if not isinstance(days, int) or days < 0:
        return None
    if days >= 60:
        return f"about {days // 30} months ({days} days)"
    if days >= 14:
        return f"about {days // 7} weeks ({days} days)"
    return f"{days} days"


def _prune(obj):
    """Drop empty values to keep the prompt small (Groq free tier: ~8k tokens/min)."""
    if isinstance(obj, dict):
        out = {k: _prune(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        return [_prune(x) for x in obj if x not in (None, "", [], {})]
    return obj


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You write one WhatsApp message for magicpin's merchant assistant "Vera" (India).
Output a JSON object: {"body": "...", "cta": "...", "rationale": "..."}.

HARD RULES (a message breaking any of these is discarded):
- Use ONLY facts in FACT_SHEET. Never invent numbers, prices, dates, names, sources, slots, competitors or research.
- Drafted plans / deliverables: EVERY number (class size, price, duration, count) must appear in FACT_SHEET. Describe anything else in words ("small batches", "you set the price").
- When quoting a digest or research fact, attribute it inline: "Per <source>: <fact>".
- Don't claim you've already drafted something and then ask to draft it: either offer ("Want me to draft...? Reply YES") or say it's ready ("I've drafted it — Reply YES to post it"), not both.
- Never add descriptive details that aren't in FACT_SHEET: no country/origin, no cause or reason for a change, no day of the week, no "free", no safety or medical claims, no guarantees.
- Never use one number with two meanings. When describing a change, state the merchant's current value (current_value_last_30d) and say which number is a peer average.
- Only offer what Vera can do: draft messages/posts/checklists/replies, set reminders, share summaries, set up an offer or post on the profile. Never claim to access or filter customer lists/records, call anyone, or take payments.
- Exactly ONE ask, stated once, in one language. Never repeat the offer in both English and Hindi.
- Vera is female: Hindi first-person forms are feminine (karungi, bhejungi, kar rahi hoon).
- Never output template placeholders like <...>, {...} or [...].
- No claims, trends or cause-effect statements that FACT_SHEET doesn't state (e.g. don't say "combos boost demand" unless a fact says so).
- Copy numbers exactly as written in FACT_SHEET (e.g. "38%", "2,100", "₹299"). Do no new arithmetic.
- No URLs. No taboo words. No preamble ("Hope you're well"). Don't introduce yourself.
- Merchant-facing to a dentist: start with the given salutation, e.g. "Dr. Meera,".
- Customer-facing: write as the business (e.g. "Dr. Meera's Dental Clinic here"), never mention Vera or magicpin; no medical claims.
- If placeholder_payload is true, the trigger has no specifics: rely on merchant/category facts only and don't pretend to know details.
- catalog_offers_suggestions_only are NOT live offers: only suggest them ("a hook like ..."), never say the merchant runs them.

WHAT SCORES WELL:
- First line says WHY NOW (the trigger), anchored on a concrete, checkable fact (number, date, source).
- Personal to THIS merchant: name the business (business_name) and use at least one of THEIR own numbers (performance, peer comparison, customer counts, reviews or live offer).
- Opportunity triggers (festival, seasonal, curious ask, spike, milestone): cite the merchant's STRONGEST own number (e.g. a positive 7-day change or an above-peer metric) as proof of demand.
- Customer-facing: name the business in the first sentence (e.g. "Apollo Health Plus Pharmacy here").
- Voice by category: dentists clinical peer; salons warm-practical; restaurants fellow operator; gyms coach-like; pharmacies trustworthy-precise.
- One engagement lever or two: loss aversion, social proof (peer numbers), curiosity, effort externalisation ("I've drafted it"), asking the merchant a question.
- Exactly ONE call to action, in the LAST sentence. Action triggers: a binary ask ending with "Reply YES" (or "Reply CONFIRM" for dispatch/booking confirmation). Curious-ask: an open question. Booking with given slots: numbered slot choice.
- language_mode "english_with_light_hindi" (merchants): English-led, clinical/technical terms in English, plus ONE natural Hindi touch in Roman script (a short phrase or the closing ask). Not mostly Hindi.
- language_mode "hinglish" (customers): natural Hindi-English code-mix in Roman script.
- language_mode "english": English only.
- Concise: 250-500 characters. Plain text, "\\n" allowed for a short list.

cta must be one of: binary_yes_stop, binary_confirm_cancel, open_ended, multi_choice_slot, none.
rationale: one sentence on which facts/levers you used and why."""

KIND_GUIDANCE = {
    "research_digest": "Lead with the source + the key finding; connect it to this merchant's patient/customer base; offer to pull the summary and draft a shareable note.",
    "regulation_change": "Compliance: state the change, the deadline and the concrete step; offer a ready checklist. Calm and precise, no fear-mongering.",
    "cde_opportunity": "Professional-development event: title, date/time, credits, fee; offer to add a reminder.",
    "supply_alert": "Urgent recall: batches, manufacturer, risk level as stated; offer to draft the customer note + replacement workflow. Don't estimate affected counts.",
    "perf_dip": "Name the dip with the exact numbers (or, if the data shows no dip, frame as a check-in with the peer gap). Offer one concrete fix.",
    "perf_spike": "Celebrate the lift with exact numbers and likely driver; offer to double down while momentum lasts.",
    "seasonal_perf_dip": "Reassure: the dip is expected (cite the category pattern). Then make it about THIS business: their active member count, their churn vs peer average (from peer_comparison), and their live offer as a retention/bring-a-friend tool. End with one retention action (e.g. a member challenge).",
    "renewal_due": "Renewal timing + amount, plus the value delivered (their real 30-day numbers). Loss aversion, not pressure.",
    "festival_upcoming": "Festival + date (+ days away if computed); category seasonal pattern; a service+price hook; offer to draft the campaign.",
    "curious_ask_due": "Ask ONE easy question about what's in demand this week, with a concrete guess from their data; promise to turn the answer into content. cta open_ended.",
    "winback_eligible": "Their plan lapsed: days since, what they're losing (numbers); offer to restart with a comeback offer.",
    "dormant_with_vera": "Re-open the conversation with one fresh, specific number about their business; low-effort ask.",
    "ipl_match_today": "Match, venue, time; use the IPL digest insight (weekend vs weeknight); give an operator's judgment on what to push tonight; check whether their offer's days apply.",
    "review_theme_emerged": "Theme + count + real quote; balance with a positive theme; offer a public reply draft + one fix.",
    "milestone_reached": "Name the milestone precisely; offer a low-effort push (e.g. thank-you note asking for reviews).",
    "active_planning_intent": "The merchant ALREADY said yes and asked what it looks like: deliver a concrete starter draft now as a short bullet list. Build it ONLY from facts: the proposal already in last_conversation_turns, their live offer, their conversion/retention vs peers, and what reviews praise. Any spec not in the facts (class size, tiers, timings) goes in words, never numbers. No qualifying questions. End with a YES to turn it into posts.",
    "category_seasonal": "State the demand shifts exactly; the recommended shelf/stock action; offer to draft a post.",
    "gbp_unverified": "Unverified profile: estimated uplift, their views vs peer, verification path; offer to walk them through it.",
    "competitor_opened": "Competitor facts only if given (name, distance, offer); don't recommend a price war; lean on their review strengths.",
    "recall_due": "Customer reminder from the business: service due + date, the given slots, a live offer if relevant. Numbered slot reply if slots exist.",
    "customer_lapsed_soft": "Warm, no-shame winback from the business; time since last visit. End with ONE concrete ask combining the live offer (if any) and their preferred slot (if known); single YES, no commitment.",
    "customer_lapsed_hard": "Warm, no-shame winback from the business; reference their past goal if known. End with ONE concrete ask that combines the live offer and their preferred slot (e.g. 'book your first free trial class on a weekday evening this week? Reply YES — no commitment').",
    "chronic_refill_due": "Refill reminder: medicines by name, run-out date, live offers (senior discount, delivery); ask to CONFIRM dispatch. If a molecule is in a recall digest item, reassure about a batch check.",
    "appointment_tomorrow": "Short reminder for tomorrow's appointment; confirm or reschedule.",
    "trial_followup": "Follow up the trial (date), offer the next session option(s), live offer if relevant; YES to hold a spot.",
    "wedding_package_followup": "Bridal follow-up: wedding date (+ days to go if computed), trial date, the next program window; offer to block their preferred slot.",
}


def build_user_prompt(sheet: dict, draft: dict) -> str:
    guidance = KIND_GUIDANCE.get(sheet["trigger"]["kind"],
                                 "Unknown trigger kind: explain plainly why you're writing now using only the facts, and make one low-effort ask.")
    return (
        f"TRIGGER GUIDANCE: {guidance}\n\n"
        f"FACT_SHEET (JSON):\n{json.dumps(sheet, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"BASELINE DRAFT (all facts verified; keep its facts, make it sharper, more specific and more compelling — "
        f"don't just copy it):\n{draft['body']}\n\n"
        f"Suggested cta: {draft['cta']}. Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def normalize(text: str) -> str:
    """Clean LLM text: every Unicode space -> normal space (never deleted, so words
    can't get glued), zero-width chars removed, lookalike hyphens -> '-', no
    double signs ("fell -30%"), and a/an before numbers."""
    text = F.clean_spaces(text)
    text = fix_double_signs(text)
    return F.fix_articles(text)


DOUBLE_SIGN = F.DOUBLE_SIGN
fix_double_signs = F.fix_double_signs


def _numbers(text: str) -> set[str]:
    """All numbers in a text, canonical form: '2,100' -> '2100', '2.10' -> '2.1'."""
    out = set()
    for tok in _NUM.findall(text or ""):
        t = tok.replace(",", "")
        try:
            v = float(t)
        except ValueError:
            continue
        out.add(f"{v:g}")
    return out


_UNIT_NUM = re.compile(
    r"(?:₹\s?(\d[\d,]*(?:\.\d+)?))|(\d[\d,]*(?:\.\d+)?)\s*(?:%|months?\b|weeks?\b|days?\b|years?\b|hours?\b|hrs?\b|"
    r"mins?\b|minutes?\b|km\b|x\b|mahine\b|hafte\b|din\b|saal\b)", re.I)


def _unit_numbers(text: str) -> set[str]:
    out = set()
    for a, b in _UNIT_NUM.findall(text or ""):
        tok = (a or b).replace(",", "")
        try:
            out.add(f"{float(tok):g}")
        except ValueError:
            pass
    return out


PLAN_KINDS = {"active_planning_intent"}


def is_drafted_plan(body: str, trigger: dict | None) -> bool:
    """A drafted plan/deliverable: a planning trigger, or any body with a bullet list."""
    return ((trigger or {}).get("kind") in PLAN_KINDS
            or bool(re.search(r"(^|\n)\s*(?:[•*\-–]|\d+[.)])\s", body or "")))


def validate(body: str, sheet: dict, draft: dict, category: dict, trigger: dict) -> list[str]:
    """Return a list of problems; empty list = OK to send."""
    problems = []
    low = body.lower()
    if not body.strip():
        return ["empty body"]
    if len(body) > MAX_BODY_CHARS:
        problems.append(f"too long ({len(body)} chars)")
    if re.search(r"https?://|www\.|\.com\b|\.in/", low):
        problems.append("contains a URL")
    for w in F.taboo_words(category):
        w_core = re.sub(r"\s*\(.*?\)\s*", "", w).strip().lower()  # 'FDA-approved (use only...)' -> 'fda-approved'
        if w_core and w_core in low:
            problems.append(f"taboo word '{w_core}'")
    # Every number in the body must exist in the verified facts (or the template draft,
    # which is built only from facts). Small counts <= 10 are allowed ("2 slots", "3-point").
    # Numbers inside taboo phrases ("shred in 7 days") are not facts: exclude that list.
    fact_part = {**sheet, "category": {k: v for k, v in (sheet.get("category") or {}).items() if k != "taboo_words"}}
    allowed = _numbers(json.dumps(fact_part, ensure_ascii=False)) | _numbers(draft["body"])
    # Drafted plans/deliverables (bullet lists, planning triggers) get NO free pass:
    # "max 6 kids per class" is a made-up spec even though 6 is small.
    strict = is_drafted_plan(body, trigger)
    invented = [n for n in _numbers(body) if n not in allowed
                and (strict or not (float(n).is_integer() and float(n) <= 10))]
    if strict:
        # Small numbers match by coincidence ("6" = a review count), so in a plan they
        # must come from plan-relevant facts: payload, chat history, offers, digest.
        plan_src = _numbers(json.dumps([sheet.get("trigger"), sheet.get("digest_item"),
                                        (sheet.get("merchant") or {}).get("last_conversation_turns"),
                                        (sheet.get("merchant") or {}).get("active_offers")],
                                       ensure_ascii=False)) | _numbers(draft["body"])
        invented += [n for n in _numbers(body) if float(n) <= 10 and n not in plan_src and n not in invented]
    # Small numbers are only free when they're plain counts ("2 slots"). Attached to a
    # unit ("8 months", "5 km", "3%", "₹9") they are claims and must be in the facts.
    invented += [n for n in _unit_numbers(body) if n not in allowed and n not in invented]
    if invented:
        problems.append(f"numbers not in facts: {', '.join(sorted(invented)[:4])}")
    if trigger.get("scope") == "customer":
        if "vera" in low or "magicpin" in low:
            problems.append("customer message mentions Vera/magicpin")
        # The customer must know who is writing: require the business's distinctive name.
        biz = (sheet.get("merchant") or {}).get("business_name") or ""
        key_words = [w for w in re.findall(r"[a-z0-9']+", biz.lower()) if len(w) > 3
                     and w not in ("clinic", "dental", "salon", "pharmacy", "fitness", "studio", "cafe", "restaurant", "health", "family")]
        if key_words and not any(w in low for w in key_words):
            problems.append("business name missing")
    elif (category.get("slug") == "dentists") and "dr." not in low[:40]:
        problems.append("dentist salutation missing")
    if "dr. dr." in low:
        problems.append("double Dr.")
    if DOUBLE_SIGN.search(body):
        problems.append("double sign (e.g. 'fell -30%')")
    problems += common_text_problems(body, json.dumps(fact_part, ensure_ascii=False) + " " + draft["body"])
    return problems


PLACEHOLDER = re.compile(r"\{\{?[^{}\n]{0,40}\}?\}|\[[^\[\]\n]{1,40}\]|<[^<>\n]{1,40}>")
WEEKDAY_WORD = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                          r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b", re.I)
# Offer/ask verbs: two of these in one message = the ask was repeated (often once in
# English and once in Hindi: "I can draft a post... draft kar doon?").
OFFER_PHRASES = re.compile(r"\b(i can draft|i can prepare|want me to|shall i|should i|can i|"
                           r"kar doon|kar du|bana doon|bana du|bhej doon|bhej du|draft karun|karun\?)", re.I)


def common_text_problems(body: str, facts_text: str) -> list[str]:
    """Checks shared by first messages and replies (word-level, not just numbers)."""
    problems = []
    low, facts_low = body.lower(), (facts_text or "").lower()
    if PLACEHOLDER.search(body):
        problems.append("unfilled placeholder")
    if F.GLUED.search(body):
        problems.append("glued words")
    # Weekday names and "free" are classic invented details: they must be in the facts.
    for day in {m.group(1).lower() for m in WEEKDAY_WORD.finditer(body)}:
        if not re.search(rf"\b{day}", facts_low):
            problems.append(f"weekday '{day}' not in facts")
    if re.search(r"\bfree\b", low) and "free" not in facts_low:
        problems.append("'free' not in facts")
    if re.search(r"\bguarantee", low):
        problems.append("guarantee claim")
    if len(OFFER_PHRASES.findall(body)) >= 2:
        problems.append("ask repeated")
    return problems
