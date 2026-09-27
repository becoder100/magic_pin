"""
templates.py — deterministic message templates, one per trigger kind.

This is the safety net behind the LLM: it must ALWAYS produce a decent,
truthful message, instantly, with no network calls.

Rules every template follows (CLAUDE.md §11–12):
- Anchor on a real fact from the contexts (numbers, dates, sources, offers).
  All facts come from facts.py; if a fact is missing the sentence is dropped,
  never invented.
- Placeholder payloads ({"placeholder": true}) -> use merchant/category data
  only (no competitor names, prices or dates that aren't in the data).
- Odd pairings (e.g. a chronic refill trigger for a dentist) -> neutral wording
  ("your regular follow-up is due") + an honest note in the rationale.
- One CTA, in the last sentence. No URLs. No preamble. Dentists get "Dr. X".
- Language: Hinglish touch for Hindi-speaking merchants when the category voice
  allows code-mix; customers follow their own language_pref.

Public entry point: render(category, merchant, trigger, customer=None, now=None) -> dict
"""

import re

import facts as F

CATEGORY_NOUN = {"dentists": "clinic", "salons": "salon", "restaurants": "restaurant",
                 "gyms": "gym", "pharmacies": "pharmacy"}
CUSTOMER_NOUN = {"dentists": "patients", "salons": "clients", "restaurants": "regulars",
                 "gyms": "members", "pharmacies": "customers"}
VISIT_NOUN = {"dentists": "visit", "salons": "visit", "restaurants": "visit",
              "gyms": "session", "pharmacies": "order"}
EMOJI = {"dentists": "🦷", "salons": "💇", "restaurants": "🍽️", "gyms": "💪", "pharmacies": "💊"}

# Which categories each customer-facing kind naturally belongs to. Anything else
# is an "odd pairing": we use neutral wording and say so in the rationale.
NATURAL_FIT = {
    "recall_due": {"dentists", "salons"},
    "chronic_refill_due": {"pharmacies"},
    "trial_followup": {"gyms", "salons"},
    "wedding_package_followup": {"salons"},
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class Ctx:
    """Everything a template needs, computed once."""

    def __init__(self, category, merchant, trigger, customer, now):
        self.cat = category or {}
        self.m = merchant or {}
        self.t = trigger or {}
        self.cust = customer
        self.now = now or None
        self.p = self.t.get("payload") or {}
        self.kind = self.t.get("kind") or "update"
        self.slug = self.cat.get("slug") or self.m.get("category_slug") or ""
        self.placeholder = F.is_placeholder(self.t)
        self.name = F.merchant_name(self.m)
        self.first = F.owner_first_name(self.m)
        self.sal = F.salutation(self.cat, self.m)
        self.lang = F.merchant_lang(self.cat, self.m)
        self.cust_lang = F.customer_lang(customer, merchant)
        self.notes: list[str] = []   # rationale notes (honest description of what we did)
        self.params: list[str] = []  # template_params (the variable parts we filled in)

    def say(self, en: str, hinglish: str) -> str:
        """Pick the merchant-facing variant for this merchant's language."""
        return hinglish if self.lang == "hinglish" else en

    def say_cust(self, en: str, hinglish: str) -> str:
        return hinglish if self.cust_lang == "hinglish" else en

    def use(self, *values):
        """Record filled-in values as template params (skip empties)."""
        self.params.extend(str(v) for v in values if v not in (None, ""))


def j(*parts) -> str:
    """Join non-empty sentence parts with single spaces."""
    return re.sub(r"\s+", " ", " ".join(p.strip() for p in parts if p and p.strip())).strip()


_ABBREV = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|St|No|vs|Rs|approx)|\b[A-Z])\.$")


def first_sentences(text: str, n: int = 1, skip_if: str | None = None) -> str:
    """First n sentences; does not split after 'Dr.', 'Mr.' or initials like 'R.'."""
    sents = []
    for piece in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        if sents and _ABBREV.search(sents[-1]):
            sents[-1] += " " + piece  # previous "sentence" ended in an abbreviation: rejoin
        elif piece:
            sents.append(piece)
    if skip_if:
        sents = [s for s in sents if skip_if not in s]
    return " ".join(sents[:n])


def possessive(name: str) -> str:
    """'Sunrise Medicos' -> "Sunrise Medicos'"; 'Studio11' -> "Studio11's"."""
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def window_words(w) -> str:
    """'7d' -> '7 days'."""
    m = re.match(r"^(\d+)\s*d$", str(w or ""))
    return f"{m.group(1)} days" if m else (F.humanize(w) or "7 days")


def nice_program(token) -> str:
    """'skin_prep_program_30day' -> '30-day skin prep program'."""
    words = F.humanize(token).split()
    lead = [w for w in words if re.fullmatch(r"\d+day", w)]
    rest = [w for w in words if not re.fullmatch(r"\d+day", w)]
    return " ".join([f"{lead[0][:-3]}-day"] + rest if lead else rest)


METRIC_LABEL = {"views": "profile views", "calls": "calls", "directions": "direction requests",
                "ctr": "CTR", "leads": "leads"}


def perf_anchor(c: Ctx, positive: bool = False) -> str:
    """One real performance fact. Default: a below-peer gap (loss aversion) if any.
    positive=True: an above-peer strength instead (for good-news messages)."""
    gaps = {g["metric"]: g for g in F.peer_gaps(c.m, c.cat)}
    if positive:
        for metric in ("calls", "views", "ctr"):
            g = gaps.get(metric)
            if g and not g["below"]:
                mine, avg = F.fmt_metric(metric, g["mine"]), F.fmt_metric(metric, g["peer"])
                c.use(mine, avg)
                if metric == "ctr":
                    return f"your CTR is {mine}, above the {avg} peer average"
                return f"{mine} {METRIC_LABEL[metric]} in the last 30 days, above the {avg} peer average"
        gaps = {}  # no strength found: fall through to raw numbers, never a negative gap
    for metric in ("calls", "ctr", "views"):
        g = gaps.get(metric)
        if g and g["below"]:
            mine, avg = F.fmt_metric(metric, g["mine"]), F.fmt_metric(metric, g["peer"])
            c.use(mine, avg)
            c.notes.append(f"peer gap on {metric}")
            if metric == "ctr":
                return f"your CTR is {mine} vs a {avg} peer average"
            return f"{METRIC_LABEL[metric]} were {mine} in the last 30 days vs a {avg} peer average"
    p = F.perf(c.m)
    views, calls = F.fmt_int(p.get("views")), F.fmt_int(p.get("calls"))
    if views and calls:
        c.use(views, calls)
        return f"your profile got {views} views and {calls} calls in the last 30 days"
    return ""


def strength_line(c: Ctx) -> str:
    """The merchant's own strongest numbers, for opportunity messages: best positive
    7-day change + best above-peer 30-day metric. '' if the data shows no strength."""
    bits = []
    d = {k: v for k, v in F.deltas(c.m).items() if k in METRIC_LABEL and v >= 0.05}
    if d:
        metric, v = max(d.items(), key=lambda kv: kv[1])
        bits.append(f"{METRIC_LABEL[metric]} up {F.fmt_pct(v)} this week")
        c.use(F.fmt_pct(v))
    above = [g for g in F.peer_gaps(c.m, c.cat) if not g["below"] and g["metric"] in ("calls", "views")]
    if above:
        g = max(above, key=lambda g: g["mine"] / g["peer"])
        mine, avg = F.fmt_metric(g["metric"], g["mine"]), F.fmt_metric(g["metric"], g["peer"])
        bits.append(f"{mine} {METRIC_LABEL[g['metric']]} in 30 days vs a {avg} peer average")
        c.use(mine, avg)
    if not bits:
        return ""
    c.notes.append("merchant's own strongest numbers as proof of demand")
    return c.say(f"Momentum is on your side: {' and '.join(bits)}.",
                 f"Momentum aapke saath hai: {' and '.join(bits)}.")


# Offers that fit a festival push, per category (checked against the merchant's live offers).
FESTIVAL_OFFER_KEYWORDS = {
    "salons": ["bridal", "spa", "makeup", "facial", "keratin"],
    "restaurants": ["thali", "combo", "family", "brunch", "party"],
    "gyms": ["month", "membership", "couple", "family"],
    "dentists": ["whitening", "cleaning", "smile"],
    "pharmacies": ["delivery", "card", "check"],
}


def yes_cta(c: Ctx, en_action: str, hi_action: str) -> str:
    """Merchant CTA: single binary YES, in the merchant's language."""
    return c.say(f"Want me to {en_action}? Reply YES.", f"{hi_action}? Reply YES.")


# ---------------------------------------------------------------------------
# Merchant-facing templates (send_as = "vera")
# ---------------------------------------------------------------------------
def t_research_digest(c: Ctx) -> dict:
    item = F.digest_for_trigger(c.cat, c.t)
    if not item:
        item = F.first_digest_of_kind(c.cat, ("research",)) or F.first_digest_of_kind(c.cat, ("trend", "tech"))
        if item:
            c.notes.append("trigger named no digest item; used this category's latest research item")
    if not item:
        return t_generic(c)
    trial = F.fmt_int(item.get("trial_n"))
    c.use(item.get("source"), item.get("title"), trial)
    noun = CUSTOMER_NOUN.get(c.slug, "customers")
    anchor = ""
    agg = c.m.get("customer_aggregate") or {}
    if item.get("patient_segment") == "high_risk_adults" and agg.get("high_risk_adult_count"):
        n = F.fmt_int(agg["high_risk_adult_count"])
        anchor = f"Directly relevant to your {n} high-risk adult {noun}."
        c.use(n)
        c.notes.append("matched item segment to merchant's high-risk cohort")
    body = j(
        f"{c.sal}, {item.get('source')} has one item worth 2 minutes: {item.get('title')}.",
        (first_sentences(item.get("summary"), 1).rstrip(".") + (f" (n = {trial} patients)." if trial else "."))
        if item.get("summary") else "",
        anchor,
        yes_cta(c, f"pull the summary and draft a short {noun[:-1]}-education WhatsApp you can share",
                f"Summary + ek {noun[:-1]}-education WhatsApp draft bana doon"),
    )
    c.notes.append(f"research digest anchored on source '{item.get('source')}'")
    return {"body": body, "cta": "binary_yes_stop"}


def t_regulation_change(c: Ctx) -> dict:
    item = F.digest_for_trigger(c.cat, c.t) or F.first_digest_of_kind(c.cat, ("compliance",))
    if not item:
        return t_generic(c)
    deadline = c.p.get("deadline_iso") or item.get("date")
    dl = F.fmt_date(deadline, with_year=True)
    days = F.days_until(deadline, c.now)
    c.use(item.get("title"), item.get("source"), dl)
    body = j(
        f"{c.sal}, compliance heads-up from the {item.get('source')}: {item.get('title')}.",
        first_sentences(item.get("summary"), 2),
        f"Suggested step: {item.get('actionable').rstrip('.')}." if item.get("actionable") else "",
        f"That's {days} days from today." if days and days > 0 else "",
        yes_cta(c, f"draft a one-page checklist for your team so you're covered{' before ' + dl if dl else ''}",
                f"Team ke liye 1-page checklist bana doon{', taaki ' + dl + ' se pehle sab ready ho' if dl else ''}"),
    )
    c.notes.append("compliance item with deadline and actionable step from the digest")
    return {"body": body, "cta": "binary_yes_stop"}


def t_cde_opportunity(c: Ctx) -> dict:
    item = F.digest_for_trigger(c.cat, c.t) or F.first_digest_of_kind(c.cat, ("cde",))
    if not item:
        return t_generic(c)
    when = F.fmt_slot(item.get("date"))
    credits = c.p.get("credits") or item.get("credits")
    fee = item.get("actionable") or F.humanize(c.p.get("fee"))
    c.use(item.get("title"), when, credits)
    body = j(
        f"{c.sal}, a CDE session worth blocking: {item.get('title')} ({item.get('source')})"
        + (f" — {when}" if when else "") + (f", {credits} CDE credits" if credits else "") + ".",
        first_sentences(item.get("summary"), 2),
        f"{fee.rstrip('.')}." if fee else "",
        yes_cta(c, "add it to your calendar and send a reminder the day before",
                "Calendar mein add karke ek din pehle reminder bhej doon"),
    )
    c.notes.append("CDE event details (date, credits, fee) from the digest")
    return {"body": body, "cta": "binary_yes_stop"}


def t_supply_alert(c: Ctx) -> dict:
    item = F.digest_for_trigger(c.cat, c.t) or F.first_digest_of_kind(c.cat, ("alert",))
    molecule = c.p.get("molecule")
    batches = [b for b in c.p.get("affected_batches") or [] if b]
    mfr = c.p.get("manufacturer")
    if not (item or molecule):
        return t_generic(c)
    chronic = F.fmt_int((c.m.get("customer_aggregate") or {}).get("chronic_rx_count"))
    c.use(molecule, ", ".join(batches), mfr, chronic)
    if molecule and batches:
        head = (f"{c.sal}, urgent: voluntary recall on {molecule} batches {', '.join(batches)}"
                + (f" by {mfr}" if mfr else "") + (f" ({item.get('source')})" if item else "") + ".")
    else:
        head = f"{c.sal}, urgent: {item.get('title')} ({item.get('source')})."
    body = j(
        head,
        first_sentences((item or {}).get("summary"), 3, skip_if="(numbers in alert)"),
        f"You have {chronic} chronic-Rx customers on file — anyone dispensed these batches should hear from you first." if chronic else "",
        yes_cta(c, "draft the customer WhatsApp note and a replacement-pickup checklist",
                "Customers ke liye WhatsApp note + replacement-pickup checklist draft kar doon"),
    )
    c.notes.append("recall details from trigger + digest; did not estimate affected-customer count (not in data)")
    return {"body": body, "cta": "binary_yes_stop"}


def _offer_cta(c: Ctx, purpose_en: str, purpose_hi: str) -> str:
    """CTA built around an offer: the merchant's live one, or a catalog suggestion."""
    offer, source = F.best_offer(c.m, c.cat)
    if offer and source == "merchant":
        c.use(offer)
        return yes_cta(c, f"push your '{offer}' in a fresh Google post {purpose_en}",
                       f"Aapka '{offer}' ek fresh Google post mein push kar doon {purpose_hi}")
    if offer:
        c.use(offer)
        c.notes.append("no live offer; suggested a catalog service+price offer")
        return yes_cta(c, f"set up '{offer}' on your profile {purpose_en}",
                       f"'{offer}' aapki profile par live kar doon {purpose_hi}")
    return yes_cta(c, f"draft a fresh Google post {purpose_en}", f"Ek fresh Google post draft kar doon {purpose_hi}")


def _profile_gaps(c: Ctx) -> str:
    sig = F.signals(c.m)
    bits = []
    if sig.get("stale_posts"):
        bits.append(f"your last Google post was {str(sig['stale_posts']).replace('d', ' days')} ago")
    if sig.get("no_active_offers") or not F.active_offers(c.m):
        bits.append("there's no active offer on your profile")
    if sig.get("unverified_gbp") or ((c.m.get("identity") or {}).get("verified") is False):
        bits.append("the profile is still unverified")
    return ("Also, " + " and ".join(bits[:2]) + ".") if bits else ""


def t_perf_dip(c: Ctx) -> dict:
    metric, delta = c.p.get("metric"), c.p.get("delta_pct")
    if metric and isinstance(delta, (int, float)):
        # State the merchant's CURRENT value, not the trigger's baseline: a bare "12"
        # next to a "12 peer average" read as one number with two meanings.
        cur_v = F.perf(c.m).get(metric)
        cur = f"{F.fmt_metric(metric, cur_v)} {METRIC_LABEL.get(metric, metric)}" if cur_v is not None else None
        pct = F.fmt_pct(abs(delta))
        c.use(metric, pct, cur)
        head = (f"{c.sal}, your {METRIC_LABEL.get(metric, metric)} dropped {pct} over the last "
                f"{window_words(c.p.get('window'))}" + (f" — you've had {cur} in the last 30 days" if cur else "") + ".")
        c.notes.append("dip numbers from trigger payload")
    else:
        # Placeholder: only claim a dip the merchant's own data shows.
        d = F.deltas(c.m)
        worst = min(d.items(), key=lambda kv: kv[1]) if d else None
        if worst and worst[1] < 0:
            pct = F.fmt_pct(abs(worst[1]))
            c.use(worst[0], pct)
            head = f"{c.sal}, your {METRIC_LABEL.get(worst[0], worst[0])} are down {pct} this week."
            c.notes.append("placeholder payload; dip taken from merchant's own 7-day delta")
        else:
            head = f"{c.sal}, quick performance check on {c.name}: {perf_anchor(c)}."
            c.notes.append("placeholder payload and merchant data shows no 7-day dip; framed as a check-in, no dip claimed")
    return {"body": j(head, _profile_gaps(c),
                      _offer_cta(c, "to pull calls back this week", "taaki is hafte calls wapas aayein")),
            "cta": "binary_yes_stop"}


def t_perf_spike(c: Ctx) -> dict:
    metric, delta = c.p.get("metric"), c.p.get("delta_pct")
    if metric and isinstance(delta, (int, float)):
        pct = F.fmt_pct(abs(delta))
        cur_v = F.perf(c.m).get(metric)
        cur = f"{F.fmt_metric(metric, cur_v)} {METRIC_LABEL.get(metric, metric)}" if cur_v is not None else None
        driver = F.humanize(c.p.get("likely_driver"))
        c.use(metric, pct, cur, driver)
        head = j(f"{c.sal}, good news: your {METRIC_LABEL.get(metric, metric)} are up {pct} over the last "
                 f"{window_words(c.p.get('window'))}" + (f" — {cur} in the last 30 days" if cur else "") + ".",
                 f"Looks driven by your {driver}." if driver else "")
    else:
        d = F.deltas(c.m)
        best = max(d.items(), key=lambda kv: kv[1]) if d else None
        if best and best[1] > 0:
            pct = F.fmt_pct(best[1])
            c.use(best[0], pct)
            head = f"{c.sal}, your {METRIC_LABEL.get(best[0], best[0])} are up {pct} this week — nice momentum for {c.name}."
            c.notes.append("placeholder payload; lift taken from merchant's own 7-day delta")
        else:
            head = f"{c.sal}, {perf_anchor(c)}."
            c.notes.append("placeholder payload; no lift visible in data, stated raw numbers only")
    return {"body": j(head, yes_cta(c, "draft a follow-up Google post while the momentum lasts",
                                    "Momentum ke beech ek follow-up Google post draft kar doon")),
            "cta": "binary_yes_stop"}


def t_seasonal_perf_dip(c: Ctx) -> dict:
    metric, delta = c.p.get("metric") or "views", c.p.get("delta_pct")
    note_months = F._month_range(F.humanize(c.p.get("season_note")))
    beat = F.beat_for_months(c.cat, note_months) or F.seasonal_beat(c.cat, F.now_month(c.now))
    members = F.fmt_int((c.m.get("customer_aggregate") or {}).get("total_active_members"))
    pct = F.fmt_pct(abs(delta)) if isinstance(delta, (int, float)) else None
    noun = CUSTOMER_NOUN.get(c.slug, "customers")
    c.use(metric, pct, members, (beat or {}).get("month_range"))
    # The merchant's own retention numbers make the "focus on retention" advice concrete:
    # e.g. PowerHouse's monthly churn 10% vs an 8% peer average.
    bench = {a["key"]: a for a in F.aggregate_vs_peer(c.m, c.cat)}
    churn = bench.get("monthly_churn_pct")
    retention_line = ""
    if members and churn:
        retention_line = (f"Where to focus: your {members} active {noun} — monthly churn is {churn['mine']} vs a "
                          f"{churn['peer']} peer average, so holding on to them is worth more than chasing new sign-ups now.")
        c.use(churn["mine"], churn["peer"])
        c.notes.append("retention action anchored on merchant's churn vs peer")
    elif members:
        retention_line = f"Where to focus: your {members} active {noun} — retention is worth more than new sign-ups in this window."
    offer = next((o for o in F.active_offers(c.m) if "trial" in o.lower() or "free" in o.lower()), None)
    offer_line = (f"Your live '{offer}' can double as a bring-a-friend pass for existing {noun}." if offer else "")
    if offer:
        c.use(offer)
    body = j(
        f"{c.sal}, {possessive(c.name)} {METRIC_LABEL.get(metric, metric)} are down {pct} this week — this is the expected "
        f"seasonal dip, not a problem with your profile." if pct else f"{c.sal}, a seasonal slowdown is showing up for {c.name}.",
        f"Category pattern for {beat['month_range']}: {beat['note']}." if beat else "",
        retention_line,
        offer_line,
        yes_cta(c, f"draft a 'stay-on-track' challenge for your {noun} to hold attendance through the dip",
                f"{noun.capitalize()} ke liye ek 'stay-on-track' challenge draft kar doon, taaki attendance bani rahe"),
    )
    c.notes.append("reframed dip as seasonal using category seasonal_beats")
    return {"body": body, "cta": "binary_yes_stop"}


def t_renewal_due(c: Ctx) -> dict:
    sub = c.m.get("subscription") or {}
    days = c.p.get("days_remaining", sub.get("days_remaining"))
    plan = c.p.get("plan") or sub.get("plan")
    amount = F.fmt_money(c.p.get("renewal_amount"))
    p = F.perf(c.m)
    views, calls, dirs = F.fmt_int(p.get("views")), F.fmt_int(p.get("calls")), F.fmt_int(p.get("directions"))
    c.use(plan, days, amount, views, calls)
    if sub.get("status") == "expired" and not c.p.get("days_remaining"):
        head = f"{c.sal}, your {plan or ''} plan lapsed {sub.get('days_since_expiry')} days ago." if sub.get("days_since_expiry") else f"{c.sal}, your plan has lapsed."
    else:
        head = (f"{c.sal}, your {plan + ' ' if plan else ''}plan renews in {days} days"
                + (f" ({amount})" if amount else "") + ".") if days is not None else f"{c.sal}, your plan is up for renewal."
    body = j(
        head,
        f"Last 30 days on your profile: {views} views, {calls} calls" + (f", {dirs} direction requests" if dirs else "") + "." if views and calls else "",
        yes_cta(c, "process the renewal so your profile upkeep doesn't pause",
                "Renewal process kar doon, taaki profile upkeep ruke nahi"),
    )
    c.notes.append("renewal reminder with the value delivered (real 30-day numbers)")
    return {"body": body, "cta": "binary_yes_stop"}


def t_festival_upcoming(c: Ctx) -> dict:
    fest = c.p.get("festival")
    date_s = c.p.get("date")
    days = F.consistent_days(c.p, "date", "days_until", c.now)
    dt = F.parse_dt(date_s)
    beat = F.seasonal_beat(c.cat, dt.month if dt else F.now_month(c.now))
    offer, source = F.best_offer(c.m, c.cat, FESTIVAL_OFFER_KEYWORDS.get(c.slug))
    c.use(fest, F.fmt_date(date_s), days if days and days > 0 else None, offer)
    if fest:
        head = f"{c.sal}, {fest} is on {F.fmt_date(date_s)}" + (f" — {days} days away" if days and days > 0 else "") + "."
    else:
        head = f"{c.sal}, festival season is coming up for {c.name}."
        c.notes.append("placeholder payload; used the category's seasonal beat for the current month")
    offer_line = ""
    if offer and source == "merchant":
        offer_line = f"Your '{offer}' is a ready hook."
    elif offer:
        offer_line = f"A service+price hook like '{offer}' tends to work better than a % discount."
        c.notes.append("no live offer; suggested catalog offer")
    label = fest or "festive"
    body = j(
        head,
        f"For {c.slug or 'your category'}, {beat['month_range']} is {beat['note']}." if beat else "",
        strength_line(c),
        offer_line,
        yes_cta(c, f"draft a {label} Google post + a WhatsApp note for your {CUSTOMER_NOUN.get(c.slug, 'customers')}",
                f"{label} ka Google post + {CUSTOMER_NOUN.get(c.slug, 'customers')} ke liye WhatsApp note draft kar doon"),
    )
    return {"body": body, "cta": "binary_yes_stop"}


CURIOUS_QUESTION = {
    "dentists": ("which treatment have patients asked about most this week?",
                 "is hafte patients ne sabse zyada kis treatment ke baare mein poocha?"),
    "salons": ("which service has been most asked-for at {name} this week?",
               "is hafte {name} par sabse zyada kaunsi service maangi gayi?"),
    "restaurants": ("which dish has been moving fastest at {name} this week?",
                    "is hafte {name} mein sabse zyada kaunsi dish bik rahi hai?"),
    "gyms": ("which class or slot has been fullest this week?",
             "is hafte kaunsi class ya slot sabse full raha?"),
    "pharmacies": ("which product have customers asked for most this week?",
                   "is hafte customers ne sabse zyada kya maanga?"),
}


def t_curious_ask_due(c: Ctx) -> dict:
    q_en, q_hi = CURIOUS_QUESTION.get(c.slug, ("what have customers asked for most this week?",
                                                "is hafte customers ne sabse zyada kya poocha?"))
    q_en, q_hi = q_en.format(name=c.name), q_hi.format(name=c.name)
    # A concrete guess makes the question easier to answer (and more specific).
    pos = sorted([r for r in c.m.get("review_themes") or [] if r.get("sentiment") == "pos"],
                 key=lambda r: -(r.get("occurrences_30d") or 0))
    offers = F.active_offers(c.m)
    guess = ""
    if offers:
        praise = f" ({pos[0]['occurrences_30d']} reviews praised {F.humanize(pos[0]['theme'])} this month)" if pos and pos[0].get("occurrences_30d") else ""
        guess = c.say(f"Is it still your {offers[0]}{praise}, or something new?",
                      f"Kya abhi bhi {offers[0]}{praise}, ya kuch naya?")
        c.use(offers[0])
    body = j(
        c.say(f"{c.sal}! Quick question — {q_en}", f"{c.sal}! Quick sawaal — {q_hi}"),
        guess,
        strength_line(c),
        c.say("Just reply in one line and I'll turn it into a Google post plus a ready WhatsApp reply for customers who ask.",
              "Bas ek line mein reply kijiye — main usse Google post + customers ke liye ready WhatsApp reply bana dungi."),
    )
    c.notes.append("curious-ask: low-effort question, reciprocity (we do the post)")
    return {"body": body, "cta": "open_ended"}


def t_winback_eligible(c: Ctx) -> dict:
    sub = c.m.get("subscription") or {}
    days = c.p.get("days_since_expiry", sub.get("days_since_expiry"))
    dip = c.p.get("perf_dip_pct")
    if dip is None:
        dip = F.deltas(c.m).get("calls")
    lapsed = F.fmt_int(c.p.get("lapsed_customers_added_since_expiry"))
    c.use(days, F.fmt_pct(abs(dip)) if isinstance(dip, (int, float)) else None, lapsed)
    body = j(
        f"{c.sal}, it's been {days} days since {possessive(c.name)} plan lapsed." if days else f"{c.sal}, {c.name}'s plan has lapsed.",
        f"Since then calls are down {F.fmt_pct(abs(dip))}" if isinstance(dip, (int, float)) and dip < 0 else "",
        (f"and {lapsed} more customers have gone inactive." if isinstance(dip, (int, float)) and dip < 0 else f"{lapsed} more customers have gone inactive since.") if lapsed else ("." if isinstance(dip, (int, float)) and dip < 0 else ""),
        yes_cta(c, "restart your profile upkeep and draft a comeback offer" + (f" for those {lapsed} customers" if lapsed else ""),
                "Profile upkeep restart karke ek comeback offer draft kar doon" + (f" un {lapsed} customers ke liye" if lapsed else "")),
    )
    body = body.replace(" .", ".")
    c.notes.append("winback: loss framing from real post-expiry numbers")
    return {"body": body, "cta": "binary_yes_stop"}


def t_dormant_with_vera(c: Ctx) -> dict:
    days = c.p.get("days_since_last_merchant_message")
    topic = F.humanize(c.p.get("last_topic"))
    c.use(days, topic)
    anchor = perf_anchor(c)
    body = j(
        (f"{c.sal}, it's been {days} days since we last spoke" + (f" (about your {topic})" if topic else "") + ".")
        if days else f"{c.sal}, it's been a while since we last spoke.",
        f"One number worth a look: {anchor}." if anchor else "",
        yes_cta(c, "send a 3-point update on what's changed on your profile",
                "Profile par kya badla, uska 3-point update bhej doon"),
    )
    if c.placeholder:
        c.notes.append("placeholder payload; re-engagement anchored on merchant's real numbers")
    return {"body": body, "cta": "binary_yes_stop"}


def t_ipl_match_today(c: Ctx) -> dict:
    match, venue = c.p.get("match"), c.p.get("venue")
    when = F.fmt_slot(c.p.get("match_time_iso"))
    time_only = when.split(", ")[-1] if when and ", " in when else None
    item = next((d for d in c.cat.get("digest") or [] if "ipl" in (d.get("title") or "").lower()), None)
    weeknight = c.p.get("is_weeknight")
    offers = F.active_offers(c.m)
    c.use(match, venue, time_only)
    head = (f"Heads-up {c.first or c.name} — {match}" + (f" at {venue}" if venue else "")
            + (f" today, {time_only}" if time_only else " today") + ".") if match else f"{c.sal}, there's an IPL match today."
    # Attribute the digest fact inline ("Per <source>: ...") so the numbers read as
    # verifiable data, and include the weekend-vs-weeknight contrast (both sentences).
    insight = f"Per {item.get('source')}: {first_sentences(item.get('summary'), 2)}" if item else ""
    advice, cta = "", None
    tue_thu = next((o for o in offers if "tue-thu" in o.lower()), None)
    if weeknight is False and tue_thu:
        advice = f"Your '{tue_thu}' doesn't cover tonight, so a delivery-only match-night push fits better."
        c.use(tue_thu)
        c.notes.append("weekend match: advised delivery push, noted Tue-Thu offer doesn't apply")
        cta = yes_cta(c, "draft a delivery-only match-night post for tonight",
                      "Aaj raat ke liye delivery-only match-night post draft kar doon")
    elif offers:
        advice = f"Your '{offers[0]}' is a natural match-night hook."
        c.use(offers[0])
    if not cta:
        cta = yes_cta(c, "draft a match-night post for tonight", "Aaj raat ke liye match-night post draft kar doon")
    return {"body": j(head, insight, advice, cta),
            "cta": "binary_yes_stop"}


def t_review_theme_emerged(c: Ctx) -> dict:
    theme, occ = c.p.get("theme"), c.p.get("occurrences_30d")
    quote, trend = c.p.get("common_quote"), c.p.get("trend")
    if not theme:  # placeholder: use the merchant's own most-mentioned negative theme
        negs = sorted([r for r in c.m.get("review_themes") or [] if r.get("sentiment") == "neg"],
                      key=lambda r: -(r.get("occurrences_30d") or 0))
        if negs:
            theme, occ, quote = negs[0].get("theme"), negs[0].get("occurrences_30d"), negs[0].get("common_quote")
            c.notes.append("placeholder payload; theme taken from merchant's review_themes")
    if not theme:
        return t_generic(c)
    pos = sorted([r for r in c.m.get("review_themes") or [] if r.get("sentiment") == "pos"],
                 key=lambda r: -(r.get("occurrences_30d") or 0))
    c.use(occ, F.humanize(theme), quote)
    body = j(
        f"{c.sal}, {occ} reviews in the last 30 days mention {F.humanize(theme)}"
        + (f", and the trend is {trend}" if trend else "") + "." if occ else f"{c.sal}, a review theme is building: {F.humanize(theme)}.",
        f"One says: \"{quote}\"." if quote else "",
        f"On the plus side, {pos[0]['occurrences_30d']} reviews praise {F.humanize(pos[0]['theme'])}." if pos and pos[0].get("occurrences_30d") else "",
        yes_cta(c, "draft a calm public reply plus one fix you can announce",
                "Ek calm public reply + ek fix jo aap announce kar sakein, draft kar doon"),
    )
    return {"body": body, "cta": "binary_yes_stop"}


def t_milestone_reached(c: Ctx) -> dict:
    metric = c.p.get("metric")
    now_v, goal = c.p.get("value_now"), c.p.get("milestone_value")
    label = "Google reviews" if metric == "review_count" else F.humanize(metric)
    if isinstance(now_v, (int, float)) and isinstance(goal, (int, float)):
        c.use(now_v, goal, label)
        if c.p.get("is_imminent") and goal > now_v:
            head = f"{c.sal}, {c.name} is at {F.fmt_int(now_v)} {label} — just {F.fmt_int(goal - now_v)} away from {F.fmt_int(goal)}."
        else:
            head = f"{c.sal}, {c.name} just crossed {F.fmt_int(goal)} {label}."
        cta = yes_cta(c, "draft a short thank-you WhatsApp asking happy regulars for a review to get you over the line this week",
                      "Happy regulars ke liye ek chhota thank-you WhatsApp draft kar doon, review maangte hue, taaki is hafte milestone cross ho")
    else:
        head = f"{c.sal}, a milestone worth marking for {c.name}: {perf_anchor(c, positive=True)}."
        c.notes.append("placeholder payload; milestone framed with merchant's real 30-day numbers")
        cta = yes_cta(c, "draft a thank-you post for your customers", "Customers ke liye ek thank-you post draft kar doon")
    return {"body": j(head, cta), "cta": "binary_yes_stop"}


PLAN_BULLETS = {
    "restaurants": ["Volume tiers by order size — you set the per-plate price for each tier",
                    "Order cut-off the evening before, so the kitchen can plan",
                    "Launch: a Google post + a WhatsApp note to nearby offices"],
    "gyms": ["Batches by age group with small class sizes",
             "Fixed weekly slots + one trial session for new sign-ups",
             "Launch: a Google post + a WhatsApp note to your members' families"],
    "salons": ["One fixed service bundle with one clear price",
               "Bookable slots on your quietest days",
               "Launch: a Google post + a WhatsApp note to regulars"],
    "dentists": ["One clearly defined service with one clear price",
                 "Dedicated weekly slots so it doesn't clash with regular appointments",
                 "Launch: a patient-education post + a WhatsApp note to your patient list"],
    "pharmacies": ["Clear eligibility and one clear price",
                   "Home-delivery option for repeat customers",
                   "Launch: a counter display + a WhatsApp note to repeat customers"],
}


def history_facts(c: Ctx) -> tuple[str | None, str | None]:
    """Facts Vera already stated in this merchant's chat history (newest first):
    (proposal, demand) e.g. ('4-week program, 3 classes/week, age 7-12, ₹2,499',
                             'Your weekday thali is doing well — 18 orders/day avg').
    Reusing them keeps a drafted plan specific without inventing any number."""
    proposal = demand = None
    for turn in reversed(c.m.get("conversation_history") or []):
        if turn.get("from") != "vera":
            continue
        text = turn.get("body") or ""
        m = re.search(r"\bSuggest(?:ed|ing)?\s*:?\s*(.+?)(?:\.\s|\.$|\?|$)", text)
        if m and not proposal and re.search(r"\d", m.group(1)):
            proposal = m.group(1).strip()
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if not demand and re.search(r"\d", sent) and "?" not in sent and "suggest" not in sent.lower():
                demand = sent.strip().rstrip(".")
    return proposal, demand


def t_active_planning_intent(c: Ctx) -> dict:
    """The merchant already said yes: deliver a draft (action mode), don't ask qualifying questions.
    Every number in the plan comes from the data: the earlier proposal in chat history,
    live offers, review counts, conversion vs peers. Unknown specs are left in words."""
    topic = F.humanize(c.p.get("intent_topic")) or "new idea"
    kws = [w for w in topic.split() if len(w) > 3]
    offer = next((o for o in F.active_offers(c.m) if any(k in o.lower() for k in kws)), None)
    agg = c.m.get("customer_aggregate") or {}
    proposal, demand = history_facts(c)
    bench = {a["key"]: a for a in F.aggregate_vs_peer(c.m, c.cat)}
    pos = sorted([r for r in c.m.get("review_themes") or [] if r.get("sentiment") == "pos" and r.get("occurrences_30d")],
                 key=lambda r: -r["occurrences_30d"])[:2]
    bullets = []
    if proposal:
        bullets.append(f"Format (as we discussed): {proposal}")
        c.use(proposal)
        c.notes.append("plan format reused from the proposal already in chat history")
    if offer:
        bullets.append(f"Built on your {offer} as the base")
        c.use(offer)
    if demand:
        bullets.append(f"Demand proof: {demand[0].lower() + demand[1:]}")
        c.use(demand)
    ttp = bench.get("trial_to_paid_pct")
    if ttp and ttp["better"]:
        bullets.append(f"Lead with one free trial session — your trial-to-paid is {ttp['mine']} vs a {ttp['peer']} peer average")
        c.use(ttp["mine"], ttp["peer"])
    if pos:
        bullets.append("Selling points customers already praise: "
                       + " and ".join(f"{F.humanize(r['theme'])} ({r['occurrences_30d']} reviews)" for r in pos))
    if len(bullets) < 3:  # thin data: fall back to generic (number-free) structure bullets
        bullets += PLAN_BULLETS.get(c.slug, ["One clear offer with one clear price", "Fixed slots"])[: 3 - len(bullets)]
    bullets.append(PLAN_BULLETS.get(c.slug, ["Launch: a Google post + a WhatsApp note to regulars"])[-1])
    extra = ""
    if agg.get("delivery_share_pct"):
        extra = f"{F.fmt_pct(agg['delivery_share_pct'])} of your orders are already delivery, so the logistics are in place."
        c.use(F.fmt_pct(agg["delivery_share_pct"]))
    c.use(topic)
    body = (f"{c.sal}, here's a starter plan for {possessive(c.name)} {topic} — edit anything:\n"
            + "\n".join(f"• {b}" for b in bullets)
            + ("\n" + extra if extra else "") + "\n"
            + c.say("Reply YES and I'll turn this into a Google post + a WhatsApp announcement draft.",
                    "YES reply karein, main isse Google post + WhatsApp announcement draft mein badal dungi."))
    c.notes.append("merchant already expressed intent -> delivered a draft plan (action mode), no qualifying question")
    return {"body": body, "cta": "binary_yes_stop"}


def t_category_seasonal(c: Ctx) -> dict:
    trends = []
    for tr in c.p.get("trends") or []:
        m = re.match(r"^(.*?)_demand_([+-]\d+)$", str(tr))
        if m:
            trends.append(f"{F.humanize(m.group(1))} {m.group(2)}%")
    season = F.humanize(c.p.get("season"))
    item = next((d for d in c.cat.get("digest") or [] if d.get("kind") == "seasonal"), None)
    if trends:
        c.use(season, *trends)
        head = f"{c.sal}, the {season} demand shift is showing up: {', '.join(trends)}."
    else:
        beat = F.seasonal_beat(c.cat, F.now_month(c.now))
        top = sorted(c.cat.get("trend_signals") or [], key=lambda s: -(s.get("delta_yoy") or 0))[:2]
        searches = "; ".join(f"'{s['query']}' searches {F.fmt_pct(s['delta_yoy'], signed=True)} YoY" for s in top if s.get("query"))
        head = j(f"{c.sal}, seasonal shift for {c.slug}:",
                 f"{beat['month_range']} — {beat['note']}." if beat else "",
                 f"Also trending: {searches}." if searches else "")
        c.notes.append("placeholder payload; used category seasonal beats + trend signals")
    body = j(
        head,
        f"Suggested move: {item['actionable'].rstrip('.')} ({item.get('source')})." if item and item.get("actionable") and trends else "",
        strength_line(c),
        yes_cta(c, f"draft a '{season or 'seasonal'} essentials' post for your profile",
                f"Aapki profile ke liye '{season or 'seasonal'} essentials' post draft kar doon"),
    )
    return {"body": body, "cta": "binary_yes_stop"}


def t_gbp_unverified(c: Ctx) -> dict:
    uplift = F.fmt_pct(c.p.get("estimated_uplift_pct")) if c.p.get("estimated_uplift_pct") else None
    path = F.humanize(c.p.get("verification_path")).replace(" or ", " or a ")
    gaps = {g["metric"]: g for g in F.peer_gaps(c.m, c.cat)}
    views = gaps.get("views")
    c.use(uplift, path)
    body = j(
        f"{c.sal}, {possessive(c.name)} Google profile is still unverified.",
        f"Verifying is estimated to lift visibility by about {uplift}." if uplift else "",
        f"Right now you get {F.fmt_int(views['mine'])} views a month vs a {F.fmt_int(views['peer'])} peer average." if views and views["below"] else "",
        f"Verification is by {path}." if path else "",
        yes_cta(c, "start the verification request and walk you through it",
                "Verification request shuru karke aapko step-by-step guide kar doon"),
    )
    return {"body": body, "cta": "binary_yes_stop"}


def t_competitor_opened(c: Ctx) -> dict:
    name, dist = c.p.get("competitor_name"), c.p.get("distance_km")
    their, opened = c.p.get("their_offer"), F.fmt_date(c.p.get("opened_date"))
    pos = sorted([r for r in c.m.get("review_themes") or [] if r.get("sentiment") == "pos"],
                 key=lambda r: -(r.get("occurrences_30d") or 0))
    if name:
        c.use(name, dist, their, opened)
        head = (f"{c.sal}, heads-up: {name} opened" + (f" {dist} km from you" if dist else " near you")
                + (f" on {opened}" if opened else "") + (f", advertising {their}" if their else "") + ".")
        kw = [w for w in re.findall(r"[A-Za-z]+", their or "") if len(w) > 4]
        mine = next((o for o in F.active_offers(c.m) if any(k.lower() in o.lower() for k in kw)), None)
        if mine:
            head += f" That undercuts your {mine}."
            c.use(mine)
    else:
        head = f"{c.sal}, a new {CATEGORY_NOUN.get(c.slug, 'business')} has opened near {F.locality(c.m) or 'you'}."
        c.notes.append("placeholder payload; no competitor name/distance/offer invented")
    strength = ""
    if pos and pos[0].get("occurrences_30d"):
        strength = (f"Rather than a price war, lean on what customers already praise: "
                    f"{pos[0]['occurrences_30d']} reviews this month mention {F.humanize(pos[0]['theme'])}.")
        c.use(pos[0]["occurrences_30d"])
    else:
        anchor = perf_anchor(c)
        strength = f"Your baseline to protect: {anchor}." if anchor else ""
    return {"body": j(head, strength, yes_cta(c, "draft a Google post that plays to that strength this week",
                                               "Is strength par ek Google post is hafte draft kar doon")),
            "cta": "binary_yes_stop"}


def t_generic(c: Ctx) -> dict:
    """Unknown kind or missing data: short, honest, anchored on real numbers."""
    anchor = perf_anchor(c)
    what = F.humanize(c.kind)
    details = [f"{F.humanize(k)}: {v}" for k, v in c.p.items()
               if k not in ("placeholder", "metric_or_topic") and isinstance(v, (str, int, float)) and not isinstance(v, bool)][:3]
    c.use(*details)
    c.notes.append(f"generic template for kind '{c.kind}'")
    return {"body": j(f"{c.sal}, flagging one thing for {c.name} today: {what}" + (f" ({'; '.join(details)})" if details else "") + ".",
                      f"For context, {anchor}." if anchor else "",
                      yes_cta(c, "look into it and send you a 3-line summary",
                              "Isko check karke 3-line summary bhej doon")),
            "cta": "binary_yes_stop"}


# ---------------------------------------------------------------------------
# Customer-facing templates (send_as = "merchant_on_behalf")
# ---------------------------------------------------------------------------
def _hello(c: Ctx) -> str:
    addr, _ = F.customer_names(c.cust)
    c.use(addr)
    if c.cust_lang == "hinglish":
        return f"Namaste {addr}," if addr else "Namaste,"
    return f"Hi {addr}," if addr else "Hi there,"


def _live_offer(c: Ctx, keywords=None) -> str | None:
    """Customer messages only mention the merchant's LIVE offers (a catalog
    suggestion would promise the customer something the shop doesn't offer)."""
    offer, source = F.best_offer(c.m, c.cat, keywords)
    if not offer or source != "merchant":
        return None
    kws = [k.lower() for k in (keywords or []) if len(k) > 3]
    if kws and not any(k in offer.lower() for k in kws):
        return None  # e.g. don't pitch "Haircut @ Rs99" in a bridal skin-prep message
    c.use(offer)
    return offer


def _odd_pairing(c: Ctx) -> bool:
    natural = NATURAL_FIT.get(c.kind)
    odd = natural is not None and c.slug not in natural
    if odd:
        c.notes.append(f"odd pairing ({c.kind} for a {c.slug} merchant): used neutral follow-up wording")
    return odd


def _last_visit(c: Ctx) -> str | None:
    return F.fmt_date(((c.cust or {}).get("relationship") or {}).get("last_visit"))


def t_recall_due(c: Ctx) -> dict:
    emoji = EMOJI.get(c.slug, "")
    odd = _odd_pairing(c)
    service = (F.humanize(c.p.get("service_due")).replace("6 month", "6-month").replace("3 month", "3-month")
               if c.p.get("service_due") and not odd else None)
    due = F.fmt_date(c.p.get("due_date"))
    # Payload labels, minus any weekday that contradicts the date ('Wed 5 Nov' is a
    # Thursday in 2026 -> '5 Nov, 6pm'). See facts.slot_label.
    slots = [F.slot_label(s) for s in c.p.get("available_slots") or []]
    slots = [s for s in slots if s][:3]
    offer = _live_offer(c, (service or "").split())
    last = _last_visit(c)
    c.use(service, due, *slots)
    pref = F.preferred_slot(c.cust)
    if service:
        what = c.say_cust(f"your {service} is due" + (f" on {due}" if due else "") + ".",
                          f"aapka {service} due hai" + (f" ({due})" if due else "") + ".")
    else:
        visit = VISIT_NOUN.get(c.slug, "visit")
        what = c.say_cust(f"it's time for your next {visit}" + (f" — your last one was on {last}" if last else "") + ".",
                          f"aapke agle {visit} ka time ho gaya hai" + (f" — pichla {last} ko tha" if last else "") + ".")
    if slots:
        slot_line = c.say_cust(f"We have {len(slots)} slot{'s' if len(slots) > 1 else ''} open: {' or '.join(slots)}.",
                               f"Aapke liye {len(slots)} slot{'s' if len(slots) > 1 else ''} ready hain: {' ya '.join(slots)}.")
        choices = ", ".join(f"{i} for {s.split(',')[0]}" for i, s in enumerate(slots, 1))
        cta_line = c.say_cust(f"Reply {choices}, or tell us a time that works.",
                              f"Reply {choices}, ya koi aur time bataiye.")
        cta = "multi_choice_slot"
    else:
        slot_line = ""
        cta_line = c.say_cust(f"Reply YES and we'll share open slots{' for ' + pref if pref else ''}.",
                              f"YES reply karein, hum{' ' + pref + ' ke' if pref else ''} open slots bhej denge.")
        cta = "binary_yes_stop"
    body = j(f"{_hello(c)} {c.name} here {emoji}", what, slot_line,
             f"{offer}." if offer else "", cta_line)
    c.notes.append("recall reminder sent as the merchant; slots/offer only from data")
    return {"body": body, "cta": cta}


def _winback_ask(c: Ctx, offer: str | None) -> str:
    """ONE concrete ask: the live offer + the customer's own preferred slot, e.g.
    'Want us to book your first of the 3 FREE Trial Classes on a weekday evening this week?'"""
    pref = F.preferred_slot(c.cust)  # e.g. 'weekday evening'
    when_en = f"on a {pref} this week" if pref else "this week"
    when_hi = f"is hafte {pref} mein" if pref else "is hafte"
    if offer:
        c.use(pref)
        c.notes.append("single ask combining the live offer and the customer's preferred slot")
        free_trial = re.search(r"\b\d+\s+free\b.*\bclass", offer, re.I)
        what_en = f"your first of the {offer}" if free_trial else f"a slot with our {offer}"
        what_hi = f"{offer} mein se pehli class" if free_trial else f"{offer} ke saath ek slot"
        return c.say_cust(f"Want us to book {what_en} {when_en}? Reply YES — no commitment.",
                          f"{what_hi} {when_hi} book kar dein? Reply YES — koi commitment nahi.")
    return c.say_cust(f"Want us to hold a slot for you {when_en}? Reply YES — no commitment.",
                      f"{when_hi.capitalize()} aapke liye slot hold kar dein? Reply YES — koi commitment nahi.")


def t_customer_lapsed(c: Ctx) -> dict:
    days = c.p.get("days_since_last_visit")
    if days is None:
        days = F.days_until(c.now, ((c.cust or {}).get("relationship") or {}).get("last_visit")) if c.now else None
    last = _last_visit(c)
    focus = F.humanize(c.p.get("previous_focus") or ((c.cust or {}).get("preferences") or {}).get("training_focus"))
    offer = _live_offer(c)
    visit = VISIT_NOUN.get(c.slug, "visit")
    sender = f"{c.first} from {c.name}" if c.first and c.slug != "dentists" else c.name
    if days and days >= 60:
        gap_en, gap_hi = f"It's been about {days // 30} months since your last {visit}", f"Aapke last {visit} ko lagbhag {days // 30} mahine ho gaye"
    elif days and days >= 14:
        gap_en, gap_hi = f"It's been about {days // 7} weeks since your last {visit}", f"Aapke last {visit} ko lagbhag {days // 7} hafte ho gaye"
    elif last:
        gap_en, gap_hi = f"We haven't seen you since {last}", f"{last} ke baad aapse mulaqat nahi hui"
    else:
        gap_en, gap_hi = "It's been a while since your last " + visit, f"Aapke last {visit} ko kaafi time ho gaya"
    c.use(sender, days, focus)
    body = j(
        _hello(c),
        c.say_cust(f"{sender} here. {gap_en} — happens to everyone, no judgment.",
                   f"{sender} se. {gap_hi} — sabke saath hota hai, koi baat nahi."),
        c.say_cust(f"If {focus} is still the goal, we can pick up right where you left off.",
                   f"Agar {focus} abhi bhi goal hai, toh wahin se shuru kar sakte hain.") if focus else "",
        _winback_ask(c, offer),
    )
    if c.placeholder:
        c.notes.append("placeholder payload; gap computed from customer's last visit")
    c.notes.append("no-shame winback, single YES CTA")
    return {"body": body, "cta": "binary_yes_stop"}


def t_chronic_refill_due(c: Ctx) -> dict:
    if _odd_pairing(c) or not c.p.get("molecule_list"):
        # Not a pharmacy (or no molecule data): generic, honest follow-up.
        last = _last_visit(c)
        offer = _live_offer(c)
        body = j(_hello(c),
                 c.say_cust(f"{c.name} here. Your regular follow-up is due" + (f" — your last visit was on {last}" if last else "") + ".",
                            f"{c.name} se. Aapka regular follow-up due hai" + (f" — pichli visit {last} ko thi" if last else "") + "."),
                 f"{offer}." if offer else "",
                 c.say_cust("Reply YES and we'll share open slots.", "YES reply karein, hum open slots bhej denge."))
        return {"body": body, "cta": "binary_yes_stop"}

    mols = [str(m) for m in c.p["molecule_list"]]
    runout = F.fmt_date(c.p.get("stock_runs_out_iso"))
    _, subj = F.customer_names(c.cust)
    senior = ((c.cust or {}).get("identity") or {}).get("senior_citizen")
    offers = F.active_offers(c.m)
    senior_offer = next((o for o in offers if "senior" in o.lower()), None) if senior else None
    delivery_offer = next((o for o in offers if "delivery" in o.lower()), None)
    saved = c.p.get("delivery_address_saved")
    # Cross-context insight: a molecule on this list is under a recall in the category digest.
    recalled = [m for m in mols for d in c.cat.get("digest") or []
                if d.get("kind") in ("alert", "recall") and m.lower() in (d.get("title") or "").lower()]
    loc = F.locality(c.m)
    c.use(", ".join(mols), runout, senior_offer, delivery_offer)
    who_en = f"{subj}'s" if subj else "Your"
    body = j(
        c.say_cust(f"Namaste — {c.name}{', ' + loc if loc else ''} here.",
                   f"Namaste — {c.name}{', ' + loc if loc else ''} se."),
        c.say_cust(f"{who_en} {len(mols)} regular medicines ({', '.join(mols)}) run out on {runout}. Same dose, same pack, ready to go.",
                   f"{subj + ' ki' if subj else 'Aapki'} {len(mols)} regular medicines ({', '.join(mols)}) {runout} ko khatam ho rahi hain. Same dose, same pack ready hai."),
        c.say_cust(f"{senior_offer} applies.", f"{senior_offer} apply hoga.") if senior_offer else "",
        c.say_cust(f"{delivery_offer}{' to your saved address' if saved else ''}.",
                   f"{delivery_offer}{' — saved address par' if saved else ''}.") if delivery_offer else "",
        c.say_cust(f"We'll double-check the {recalled[0]} batch against the current recall before dispatch.",
                   f"{recalled[0].capitalize()} ka batch current recall ke against check karke hi bhejenge.") if recalled else "",
        c.say_cust("Reply CONFIRM to dispatch, or tell us if the dose has changed.",
                   "Dispatch ke liye CONFIRM reply karein, ya dose mein koi change ho toh bataiye."),
    )
    if recalled:
        c.notes.append(f"cross-context: {recalled[0]} is under a digest recall -> reassuring batch check")
    c.notes.append("refill reminder with molecules, run-out date and live offers; no invented total")
    return {"body": body, "cta": "binary_confirm_cancel"}


def t_appointment_tomorrow(c: Ctx) -> dict:
    when = F.fmt_slot(c.p.get("slot_iso") or c.p.get("appointment_iso") or c.p.get("time_iso"))
    service = F.humanize(c.p.get("service"))
    time_part = when.split(", ")[-1] if when and ", " in when else None
    c.use(service, time_part)
    body = j(
        _hello(c),
        c.say_cust(f"{c.name} here — a quick reminder that your {service + ' ' if service else ''}appointment is tomorrow{' at ' + time_part if time_part else ''}.",
                   f"{c.name} se — yaad dila rahe hain, kal aapka {service + ' ' if service else ''}appointment hai{', ' + time_part + ' baje' if time_part else ''}."),
        c.say_cust("Reply YES to confirm, or tell us if you need a different time.",
                   "Confirm karne ke liye YES reply karein, ya time badalna ho toh bataiye."),
    )
    if c.placeholder:
        c.notes.append("placeholder payload; no time/service invented")
    return {"body": body, "cta": "binary_confirm_cancel"}


def t_trial_followup(c: Ctx) -> dict:
    _odd_pairing(c)
    addr, subj = F.customer_names(c.cust)
    trial = F.fmt_date(c.p.get("trial_date"))
    options = [F.slot_label(o) for o in c.p.get("next_session_options") or []]
    options = [o for o in options if o][:3]
    offer = _live_offer(c)
    who = subj if subj and subj != addr else "you"
    c.use(trial, *options)
    sender = f"{c.first} from {c.name}" if c.first else c.name
    body = j(
        _hello(c),
        c.say_cust(f"{sender} here. Hope {who} enjoyed the trial" + (f" on {trial}" if trial else "") + "!",
                   f"{sender} se. Umeed hai {who if who != 'you' else 'aapko'} trial" + (f" ({trial})" if trial else "") + " pasand aaya!"),
        c.say_cust(f"Next session: {options[0]}.", f"Agla session: {options[0]}.") if len(options) == 1 else
        (c.say_cust(f"Next sessions: {' or '.join(options)}.", f"Agle sessions: {' ya '.join(options)}.") if options else ""),
        c.say_cust(f"{offer} for those who join.", f"Join karne par {offer}.") if offer else "",
        c.say_cust("Reply YES to hold a spot" + (f" for {subj}" if who != "you" else "") + ".",
                   "Spot hold karne ke liye YES reply karein."),
    )
    return {"body": body, "cta": "binary_yes_stop"}


def t_wedding_package_followup(c: Ctx) -> dict:
    _odd_pairing(c)
    wedding = F.fmt_date(c.p.get("wedding_date") or ((c.cust or {}).get("preferences") or {}).get("wedding_date"))
    days = F.consistent_days(c.p, "wedding_date", "days_to_wedding", c.now)
    trial = F.fmt_date(c.p.get("trial_completed"))
    program = nice_program(c.p.get("next_step_window_open")) if c.p.get("next_step_window_open") else None
    pref = (F.preferred_slot(c.cust) or "").title() or None
    offer = _live_offer(c, ["bridal", "skin", "facial"])
    sender = f"{c.first} from {c.name}" if c.first else c.name
    c.use(wedding, days if days and days > 0 else None, trial, program)
    body = j(
        f"{_hello(c).rstrip(',')} 💍",
        c.say_cust(f"{sender} here. Your wedding is on {wedding}" + (f" — {days} days to go" if days and days > 0 else "") + ".",
                   f"{sender} se. Aapki shaadi {wedding} ko hai" + (f" — {days} din baaki" if days and days > 0 else "") + "."),
        c.say_cust(f"Since your bridal trial on {trial}, this is the right window to start the {program}.",
                   f"{trial} ke bridal trial ke baad, {program} shuru karne ka yahi sahi time hai.") if trial and program else "",
        f"{offer}." if offer else "",
        c.say_cust(f"Want us to block a {pref} slot for your first session? Reply YES.",
                   f"Pehle session ke liye {pref} slot block kar dein? Reply YES.") if pref else
        c.say_cust("Want us to block a slot for your first session this week? Reply YES.",
                   "Pehle session ke liye is hafte slot block kar dein? Reply YES."),
    )
    return {"body": body, "cta": "binary_yes_stop"}


def t_generic_customer(c: Ctx) -> dict:
    offer = _live_offer(c)
    c.notes.append(f"generic customer template for kind '{c.kind}'")
    body = j(_hello(c),
             c.say_cust(f"{c.name} here with a quick note about your next {VISIT_NOUN.get(c.slug, 'visit')}.",
                        f"{c.name} se, aapke agle {VISIT_NOUN.get(c.slug, 'visit')} ke baare mein ek chhota note."),
             f"{offer}." if offer else "",
             c.say_cust("Reply YES and we'll set it up for you.", "YES reply karein, hum sab set kar denge."))
    return {"body": body, "cta": "binary_yes_stop"}


# ---------------------------------------------------------------------------
# Registry + entry point
# ---------------------------------------------------------------------------
MERCHANT_TEMPLATES = {
    "research_digest": t_research_digest,
    "regulation_change": t_regulation_change,
    "cde_opportunity": t_cde_opportunity,
    "supply_alert": t_supply_alert,
    "perf_dip": t_perf_dip,
    "perf_spike": t_perf_spike,
    "seasonal_perf_dip": t_seasonal_perf_dip,
    "renewal_due": t_renewal_due,
    "festival_upcoming": t_festival_upcoming,
    "curious_ask_due": t_curious_ask_due,
    "winback_eligible": t_winback_eligible,
    "dormant_with_vera": t_dormant_with_vera,
    "ipl_match_today": t_ipl_match_today,
    "review_theme_emerged": t_review_theme_emerged,
    "milestone_reached": t_milestone_reached,
    "active_planning_intent": t_active_planning_intent,
    "category_seasonal": t_category_seasonal,
    "gbp_unverified": t_gbp_unverified,
    "competitor_opened": t_competitor_opened,
}

CUSTOMER_TEMPLATES = {
    "recall_due": t_recall_due,
    "customer_lapsed_soft": t_customer_lapsed,
    "customer_lapsed_hard": t_customer_lapsed,
    "chronic_refill_due": t_chronic_refill_due,
    "appointment_tomorrow": t_appointment_tomorrow,
    "trial_followup": t_trial_followup,
    "wedding_package_followup": t_wedding_package_followup,
}


def render(category: dict, merchant: dict, trigger: dict,
           customer: dict | None = None, now: str | None = None) -> dict:
    """Deterministic message for any trigger. Never raises, never returns an empty body."""
    category, merchant = F.clean(category or {}), F.clean(merchant or {})
    trigger, customer = F.clean(trigger or {}), F.clean(customer) if customer else None
    c = Ctx(category, merchant, trigger, customer, now)
    customer_facing = trigger.get("scope") == "customer"

    fn = (CUSTOMER_TEMPLATES if customer_facing else MERCHANT_TEMPLATES).get(c.kind)
    if fn is None:
        fn = t_generic_customer if customer_facing else t_generic
    try:
        out = fn(c)
    except Exception as e:  # a template bug must never take down the tick
        c.notes.append(f"template error ({type(e).__name__}); used generic")
        out = (t_generic_customer if customer_facing else t_generic)(c)

    if customer_facing and customer is None:
        c.notes.append("no CustomerContext pushed: addressed generically, no customer details assumed")
    if customer_facing:
        c.notes.append(f"customer language: {c.cust_lang}")
    else:
        c.notes.append(f"merchant language: {c.lang}")

    kind_slug = re.sub(r"[^a-z0-9_]", "", c.kind.lower()) or "generic"
    known = kind_slug in (CUSTOMER_TEMPLATES if customer_facing else MERCHANT_TEMPLATES)
    return {
        "body": F.fix_articles(F.fix_double_signs(F.clean_spaces(out["body"]))).strip(),
        "cta": out.get("cta", "open_ended"),
        "send_as": "merchant_on_behalf" if customer_facing else "vera",
        "suppression_key": trigger.get("suppression_key") or f"{c.kind}:{trigger.get('id')}",
        "rationale": f"Template for '{c.kind}': " + "; ".join(dict.fromkeys(c.notes)) + ".",
        "template_name": f"{'merchant' if customer_facing else 'vera'}_{kind_slug if known else 'generic'}_v1",
        "template_params": list(dict.fromkeys(c.params))[:8],
    }
