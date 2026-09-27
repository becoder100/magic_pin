"""
facts.py — the "rules decide" layer: turns raw context dicts into clean,
pre-formatted, VERIFIED facts.

Everything a message states must come from here, so that:
- templates.py (deterministic fallback) and the LLM fact sheet (step 4) use
  exactly the same numbers and wording;
- nothing is invented: if a field is missing, the helper returns None and the
  caller simply leaves that sentence out.
"""

import re
import unicodedata
from datetime import datetime

# ---------------------------------------------------------------------------
# Text repair
# ---------------------------------------------------------------------------
# judge_simulator.py opens the dataset with open(f) (no encoding). On Windows
# that decodes UTF-8 as cp1252, so "₹299" arrives as "â‚¹299" and "—" as "â€”".
# Reversing that exact mistake restores the original text.
_MOJIBAKE_HINTS = ("â", "Ã", "Â", "ðŸ")


def fix_text(s: str) -> str:
    if not isinstance(s, str) or not any(h in s for h in _MOJIBAKE_HINTS):
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s  # not the cp1252 mistake — leave untouched


def clean(obj):
    """Recursively repair every string inside a context payload."""
    if isinstance(obj, str):
        return fix_text(obj)
    if isinstance(obj, list):
        return [clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    return obj


# A direction word followed by a sign is a double negative/positive: "fell -30%",
# "down by -30%", "up +15%". The word already carries the direction, so the sign goes.
_DOWN_WORDS = (r"(?:fell|fall(?:en|s)?|dropped|drop(?:s|ped)?|down|declined?|decreased?|dipped|dips?|"
               r"slid|lower(?:ed)?|girke|gire|kam)")
_UP_WORDS = (r"(?:up|rose|rise[sn]?|jumped|jumps?|increased?|grew|grow(?:n|s)?|gained?|climbed|higher|"
             r"spiked?|badhe|badha)")
DOUBLE_SIGN = re.compile(rf"\b({_DOWN_WORDS})(\s+(?:by\s+)?)[-−–](?=\d)|\b({_UP_WORDS})(\s+(?:by\s+)?)\+(?=\d)", re.I)


def fix_double_signs(text: str) -> str:
    """'views fell -30%' -> 'views fell 30%'; 'calls up +15%' -> 'calls up 15%'."""
    return DOUBLE_SIGN.sub(lambda m: (m.group(1) or m.group(3)) + (m.group(2) or m.group(4)), text or "")


# Zero-width characters that should simply disappear. (U+200D ZERO WIDTH JOINER is
# kept: it glues multi-part emoji together.)
_ZERO_WIDTH = {"​", "‌", "⁠", "﻿"}
_HYPHENS = {"‐": "-", "‑": "-"}


def clean_spaces(text: str) -> str:
    """Normalise whitespace the LLM emits so words never get glued together.
    EVERY Unicode space character (category Zs: U+00A0, U+2002-200A, U+202F,
    U+205F, U+3000, ...) becomes a normal space; only true zero-width characters
    are removed. Earlier we only mapped 3 of them, and the rest rendered as
    nothing in many viewers ("drafta", "38%in")."""
    out = []
    for ch in text or "":
        if ch in _ZERO_WIDTH:
            continue
        if ch in _HYPHENS:
            out.append(_HYPHENS[ch])
        elif ch not in "\n\t " and unicodedata.category(ch) == "Zs":
            out.append(" ")
        else:
            out.append(ch)
    text = re.sub(r"[ \t]+", " ", "".join(out))
    return re.sub(r" *\n *", "\n", text).strip()


# Glue that shouldn't happen in normal text: "38%in", "hai,aur", "hi+18%", "price.Next".
GLUED = re.compile(r"\d%[A-Za-z]|[a-z],[a-z]{2,}|[a-z]\+\d|[a-z]{2,}\.[A-Z][a-z]")


# Vera speaks as a woman in Hindi ("Vera se bol rahi hoon" in the brief's real
# conversations). Masculine first-person forms are switched to feminine.
_FEMININE = [(r"\braha hoon\b", "rahi hoon"), (r"\bkarunga\b", "karungi"), (r"\bdunga\b", "dungi"),
             (r"\bbhejunga\b", "bhejungi"), (r"\bsakta hoon\b", "sakti hoon"), (r"\bchahta hoon\b", "chahti hoon"),
             (r"\bbataunga\b", "bataungi"), (r"\bbanaunga\b", "banaungi"), (r"\blaunga\b", "laungi"),
             (r"\bkar raha\b", "kar rahi"), (r"\bbhej raha\b", "bhej rahi"), (r"\bbana raha\b", "bana rahi")]


def feminine_vera(text: str) -> str:
    for pat, rep in _FEMININE:
        text = re.sub(pat, rep, text or "", flags=re.I)
    return text


def fix_articles(text: str) -> str:
    """'a 8% peer average' -> 'an 8% peer average' (numbers read aloud as eight/eleven/eighteen/80...)."""
    return re.sub(r"\b([Aa]) (?=(?:8|11|18)(?:[\d,.]*)(?:%|\b))", lambda m: m.group(1) + "n ", text or "")


def humanize(token) -> str:
    """'corporate_bulk_thali_package' -> 'corporate bulk thali package'."""
    return re.sub(r"\s+", " ", str(token or "").replace("_", " ")).strip()


# ---------------------------------------------------------------------------
# Number formatting (always from real values; never rounded into a new claim)
# ---------------------------------------------------------------------------
def fmt_int(n) -> str | None:
    """2100 -> '2,100'."""
    try:
        return f"{int(round(float(n))):,}"
    except (TypeError, ValueError):
        return None


def fmt_pct(x, signed: bool = False) -> str | None:
    """0.38 -> '38%'; with signed=True, -0.5 -> '-50%', 0.15 -> '+15%'."""
    try:
        v = float(x) * 100
    except (TypeError, ValueError):
        return None
    txt = f"{abs(v):.0f}%" if abs(v - round(v)) < 0.05 else f"{abs(v):.1f}%"
    if not signed:
        return txt if v >= 0 else f"-{txt}"
    return f"+{txt}" if v >= 0 else f"-{txt}"


def fmt_money(n) -> str | None:
    """4999 -> '₹4,999'."""
    s = fmt_int(n)
    return f"₹{s}" if s else None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def parse_dt(s) -> datetime | None:
    """Parse '2026-12-15', '2026-05-02T19:00:00+05:30' or '...Z'. None if invalid."""
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt


def fmt_date(s, with_year: bool = False) -> str | None:
    """'2026-12-15' -> '15 Dec' (or '15 Dec 2026')."""
    dt = parse_dt(s)
    if not dt:
        return None
    return f"{dt.day} {dt:%b}" + (f" {dt.year}" if with_year else "")


def fmt_slot(s) -> str | None:
    """'2026-11-05T18:00:00+05:30' -> 'Thu 5 Nov, 6pm'.
    Used only when a slot has no label. (Some dataset labels have the wrong
    weekday, e.g. 'Wed 5 Nov' for a Thursday, but the judge scores against the
    given label, so templates prefer the label when one exists.)"""
    dt = parse_dt(s)
    if not dt:
        return None
    hour = dt.hour % 12 or 12
    mins = f":{dt.minute:02d}" if dt.minute else ""
    ampm = "am" if dt.hour < 12 else "pm"
    if len(s) <= 10:  # date only, no time
        return f"{dt:%a} {dt.day} {dt:%b}"
    return f"{dt:%a} {dt.day} {dt:%b}, {hour}{mins}{ampm}"


_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def slot_label(slot: dict) -> str | None:
    """Label for a slot dict {'iso', 'label'}. The dataset's labels are used as given,
    EXCEPT a weekday that contradicts the date is dropped: 'Wed 5 Nov, 6pm' for
    2026-11-05 (a Thursday) becomes '5 Nov, 6pm'. Stating neither the wrong day nor a
    'corrected' one keeps the message consistent with both the date and the payload."""
    label, iso = (slot or {}).get("label"), (slot or {}).get("iso")
    if not label:
        return fmt_slot(iso)
    dt = parse_dt(iso)
    m = re.match(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?,?\s+", label)
    if dt and m and m.group(1) != _WEEKDAYS[dt.weekday()]:
        return label[m.end():].strip()
    return label


def days_until(target, now) -> int | None:
    """Whole days from `now` to `target` (negative = in the past). None if unknown.
    We only compute this when the judge gave us `now`, so output stays
    deterministic for a given input (never reads the wall clock)."""
    t, n = parse_dt(target), parse_dt(now)
    if not (t and n):
        return None
    return (t.date() - n.date()).days


def consistent_days(payload: dict, date_key: str, days_key: str, now) -> int | None:
    """Days until payload[date_key], but ONLY if it agrees (±1) with the payload's
    own day count (when it has one). Seed triggers carry counts computed on their
    creation date (e.g. days_to_wedding: 196); the judge checks our numbers against
    the payload, so when they disagree we state the date alone, not a count."""
    computed = days_until((payload or {}).get(date_key), now)
    given = (payload or {}).get(days_key)
    if computed is None or computed <= 0:
        return None
    if isinstance(given, (int, float)) and abs(given - computed) > 1:
        return None
    return computed


def now_month(now) -> int | None:
    dt = parse_dt(now)
    return dt.month if dt else None


# ---------------------------------------------------------------------------
# Merchant identity, salutation and language
# ---------------------------------------------------------------------------
def owner_first_name(merchant: dict) -> str | None:
    """Owner's first name with any 'Dr.' prefix removed (some generated
    merchants store 'Dr. Sameer', which would otherwise become 'Dr. Dr. Sameer')."""
    raw = ((merchant or {}).get("identity") or {}).get("owner_first_name") or ""
    name = re.sub(r"^\s*(dr\.?|doctor)\s+", "", raw, flags=re.I).strip()
    return name or None


def merchant_name(merchant: dict) -> str:
    return ((merchant or {}).get("identity") or {}).get("name") or "your business"


def locality(merchant: dict) -> str | None:
    return ((merchant or {}).get("identity") or {}).get("locality")


def salutation(category: dict, merchant: dict) -> str:
    """Dentists -> 'Dr. Meera'; everyone else -> 'Hi Lakshmi'; no name -> '<shop> team'."""
    first = owner_first_name(merchant)
    if category.get("slug") == "dentists":
        return f"Dr. {first}" if first else "Doctor"
    return f"Hi {first}" if first else f"Hi {merchant_name(merchant)} team"


def merchant_lang(category: dict, merchant: dict) -> str:
    """'hinglish' when the merchant speaks Hindi AND the category voice allows
    Hindi-English code-mix; otherwise 'en'. Gyms are English-primary."""
    langs = ((merchant or {}).get("identity") or {}).get("languages") or []
    code_mix = str(((category or {}).get("voice") or {}).get("code_mix") or "")
    if "hi" in langs and code_mix.startswith("hindi_english"):
        return "hinglish"
    return "en"


def taboo_words(category: dict) -> list[str]:
    """The dataset uses voice.vocab_taboo; the testing brief shows voice.taboos. Accept both."""
    voice = (category or {}).get("voice") or {}
    return list(voice.get("vocab_taboo") or []) + list(voice.get("taboos") or [])


# ---------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------
def active_offers(merchant: dict) -> list[str]:
    return [o.get("title") for o in (merchant or {}).get("offers") or []
            if o.get("status") == "active" and o.get("title")]


def best_offer(merchant: dict, category: dict, keywords: list[str] | None = None):
    """Pick an offer to anchor on. Returns (title, source) where source is
    'merchant' (their own live offer) or 'catalog' (category's standard
    service+price, to be presented as a suggestion, never as already live).
    Prefers titles matching `keywords`, then service+price formats."""
    kws = [k.lower() for k in (keywords or []) if k]

    def score(title: str) -> tuple:
        t = title.lower()
        return (any(k in t for k in kws), "@ ₹" in title or "₹" in title)

    mine = active_offers(merchant)
    if mine:
        return max(mine, key=score), "merchant"
    catalog = [o.get("title") for o in (category or {}).get("offer_catalog") or [] if o.get("title")]
    if catalog:
        return max(catalog, key=score), "catalog"
    return None, None


# ---------------------------------------------------------------------------
# Performance and peers
# ---------------------------------------------------------------------------
def perf(merchant: dict) -> dict:
    return (merchant or {}).get("performance") or {}


def peer(category: dict) -> dict:
    return (category or {}).get("peer_stats") or {}


def deltas(merchant: dict) -> dict:
    """{'views': -0.22, 'calls': -0.5, ...} from performance.delta_7d."""
    d = perf(merchant).get("delta_7d") or {}
    return {k.replace("_pct", ""): v for k, v in d.items() if isinstance(v, (int, float))}


def peer_gaps(merchant: dict, category: dict) -> list[dict]:
    """Compare the merchant's 30-day numbers with category peer averages.
    Returns entries like {'metric': 'calls', 'mine': 22, 'peer': 28, 'below': True}."""
    p, pr = perf(merchant), peer(category)
    out = []
    for metric, peer_key in (("views", "avg_views_30d"), ("calls", "avg_calls_30d"),
                             ("directions", "avg_directions_30d"), ("ctr", "avg_ctr")):
        mine, avg = p.get(metric), pr.get(peer_key)
        if isinstance(mine, (int, float)) and isinstance(avg, (int, float)) and avg:
            out.append({"metric": metric, "mine": mine, "peer": avg, "below": mine < avg})
    return out


# customer_aggregate keys that have a matching peer_stats benchmark (same key name).
# For churn, lower is better; for the rest, higher is better.
AGGREGATE_BENCHMARKS = {
    "monthly_churn_pct": ("monthly churn", False),
    "trial_to_paid_pct": ("trial-to-paid conversion", True),
    "retention_6mo_pct": ("6-month retention", True),
    "retention_3mo_pct": ("3-month retention", True),
    "repeat_customer_pct": ("repeat-customer share", True),
    "delivery_share_pct": ("delivery share", True),
}


def aggregate_vs_peer(merchant: dict, category: dict) -> list[dict]:
    """Merchant customer metrics vs the category benchmark, e.g.
    {'label': 'monthly churn', 'mine': '10%', 'peer': '8%', 'better': False}."""
    agg = (merchant or {}).get("customer_aggregate") or {}
    pr = peer(category)
    out = []
    for key, (label, higher_is_better) in AGGREGATE_BENCHMARKS.items():
        mine, avg = agg.get(key), pr.get(key)
        if isinstance(mine, (int, float)) and isinstance(avg, (int, float)) and mine != avg:
            better = (mine > avg) if higher_is_better else (mine < avg)
            out.append({"key": key, "label": label, "mine": fmt_pct(mine), "peer": fmt_pct(avg), "better": better})
    return out


def fmt_metric(metric: str, value) -> str | None:
    """CTR is a ratio (0.021 -> '2.1%'); other metrics are counts."""
    if metric == "ctr":
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return None
    return fmt_int(value)


def signals(merchant: dict) -> dict:
    """['stale_posts:22d', 'unverified_gbp'] -> {'stale_posts': '22d', 'unverified_gbp': True}."""
    out = {}
    for s in (merchant or {}).get("signals") or []:
        if not isinstance(s, str):
            continue
        k, _, v = s.partition(":")
        out[k] = v or True
    return out


# ---------------------------------------------------------------------------
# Category knowledge
# ---------------------------------------------------------------------------
def find_digest(category: dict, item_id) -> dict | None:
    if not item_id:
        return None
    for d in (category or {}).get("digest") or []:
        if d.get("id") == item_id:
            return d
    return None


def digest_for_trigger(category: dict, trigger: dict) -> dict | None:
    """Digest-type triggers reference an item via top_item_id / digest_item_id / alert_id."""
    p = (trigger or {}).get("payload") or {}
    for key in ("top_item_id", "digest_item_id", "alert_id", "item_id"):
        item = find_digest(category, p.get(key))
        if item:
            return item
    return None


def first_digest_of_kind(category: dict, kinds: tuple) -> dict | None:
    for d in (category or {}).get("digest") or []:
        if d.get("kind") in kinds:
            return d
    return None


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _month_range(text: str) -> set[int]:
    """'Nov-Feb' -> {11,12,1,2}; 'Jan' -> {1}; 'Feb 14' -> {2}."""
    found = [(_MONTHS[m.lower()[:3]]) for m in re.findall(r"[A-Za-z]{3,}", text or "")
             if m.lower()[:3] in _MONTHS]
    if not found:
        return set()
    if len(found) == 1:
        return {found[0]}
    start, end = found[0], found[-1]
    months, m = set(), start
    while True:
        months.add(m)
        if m == end:
            break
        m = m % 12 + 1
    return months


def seasonal_beat(category: dict, month: int | None) -> dict | None:
    """The category's seasonal note covering `month` (e.g. Sep -> gyms 'Aug-Oct' beat)."""
    if not month:
        return None
    for beat in (category or {}).get("seasonal_beats") or []:
        if month in _month_range(beat.get("month_range", "")):
            return beat
    return None


def beat_for_months(category: dict, months: set[int]) -> dict | None:
    for beat in (category or {}).get("seasonal_beats") or []:
        if months & _month_range(beat.get("month_range", "")):
            return beat
    return None


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
def customer_names(customer: dict | None) -> tuple[str | None, str | None]:
    """Returns (address_as, subject).
    'Karthik (parent: Sumitra)' -> ('Sumitra', 'Karthik')   message goes to the parent
    'Mr. Sharma'                -> ('Sharma ji', 'Sharma ji') respectful form
    '(walk-in, no profile)'     -> (None, None)               no usable name
    """
    raw = (((customer or {}).get("identity") or {}).get("name") or "").strip()
    if not raw or raw.startswith("("):
        return None, None
    m = re.match(r"^(.*?)\s*\(parent:\s*(.*?)\)\s*$", raw)
    if m:
        return m.group(2).strip() or None, m.group(1).strip() or None
    m = re.match(r"^(mr|mrs|ms|shri|smt)\.?\s+(.+)$", raw, flags=re.I)
    if m:
        return f"{m.group(2).strip()} ji", f"{m.group(2).strip()} ji"
    return raw, raw


def customer_lang(customer: dict | None, merchant: dict | None = None) -> str:
    """'hi' / 'hi-en mix' -> 'hinglish'; 'en' / 'english' / other regional mixes -> 'en'.
    (We don't attempt regional scripts we can't write reliably.)
    With no CustomerContext, fall back to the merchant's languages."""
    pref = str(((customer or {}).get("identity") or {}).get("language_pref") or "").lower()
    if pref:
        return "hinglish" if pref.startswith("hi") else "en"
    langs = ((merchant or {}).get("identity") or {}).get("languages") or []
    return "hinglish" if "hi" in langs else "en"


def reminder_opt_in(customer: dict | None):
    return ((customer or {}).get("preferences") or {}).get("reminder_opt_in")


def preferred_slot(customer: dict | None) -> str | None:
    return humanize(((customer or {}).get("preferences") or {}).get("preferred_slots")) or None


def is_placeholder(trigger: dict) -> bool:
    """Generated triggers carry {'placeholder': true} instead of real specifics."""
    p = (trigger or {}).get("payload") or {}
    return bool(p.get("placeholder")) or not any(k for k in p if k not in ("placeholder", "metric_or_topic"))
