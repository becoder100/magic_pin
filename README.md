# Vera bot — magicpin AI Challenge (Rohit Kumar)

A backend-only FastAPI service that writes proactive WhatsApp messages for merchants (and for their customers) and handles multi-turn replies. Endpoints: `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`, `/v1/teardown`.

## Approach: "rules decide, AI writes"
1. **Rules decide what to send** (`bot.py`). Each tick picks triggers by urgency, with at most one merchant-facing message per merchant. It skips suppressed keys, merchants who opted out, and customers with `reminder_opt_in: false`. Expired triggers are not dropped: seed triggers expire in May 2026, and the harness runs on the real clock.
2. **Facts** (`facts.py`, `composer.build_fact_sheet`). A small sheet of verified facts only, pre-formatted (`38%`, `2,100`, `₹299`). It holds the salutation (`Dr. Meera`), language mode, live offers vs catalog suggestions, peer comparisons, the resolved digest item, and pre-computed date gaps. Customer messages get only customer-safe facts: no CTR, no magicpin plan, no chat history.
3. **Template draft** (`templates.py`). One deterministic template per trigger kind (26 kinds), plus generic fallbacks. It handles placeholder payloads (no invented specifics), odd pairings (neutral wording, noted in the rationale), Hinglish, and the merchant's strongest own numbers.
4. **LLM rewrite.** `gpt-oss-120b` on Groq, temperature 0, JSON mode. It receives the fact sheet, per-kind guidance and the template draft as a baseline.
5. **Validator.** It rejects the LLM's output if it contains:
   - a number not in the facts: strict for drafted plans, and for any number attached to a unit ("8 months", "5 km")
   - a taboo word
   - a URL
   - a missing `Dr.` for a dentist
   - a customer message that mentions Vera or omits the business name
   - an unfilled placeholder

   On any failure, the template is sent. The rationale always says which path produced the message.
6. **Replies** (`conversation_handlers.py`). Rule-based classification comes first, in this order:
   - **Opt-out / hostile:** `end`, and never message proactively again.
   - **Auto-reply:** counted per *merchant*, since the harness reuses the same text across conversation IDs. First: a note for the owner. Second: wait 24h. Third: end.
   - **Commitment:** action mode. Replies start with "Done…"/"Here's…", end with "Reply CONFIRM", and are checked for qualifying phrases.
   - **Later / busy:** wait.
   - **Off-topic (GST…):** decline in one line, then steer back.
   - **Soft no:** end.
   - **Anything else:** an LLM answer grounded in the facts, or a template.

   Language is detected on every turn, and a body is never repeated within a conversation.

## Rate-limit fallback design (Groq free tier)
Groq's free tier allows about 8k tokens per minute *per model*, and reserves the requested `max_tokens` up front. So:
- **Small calls:** prompts are about 1k tokens, with `max_tokens` of 700 for messages and 500 for replies.
- **Failover:** on a 429, the model cools down until its `retry-after` time, and the *same call* moves to `qwen/qwen3.8-27b`, which has its own budget. A 429 returns instantly, so this costs no time.
- **Templates last:** when both models are cooling down, the call is skipped and the template is used. `/v1/tick` composes in parallel under an 11s deadline, so it always answers inside the harness's 15s limit.
- **Offline:** `generate_submission.py` waits out cooldowns, since it has no deadline.

## Trade-offs
- Validating numbers catches invented figures, but not a wrong *meaning* of a correct number (e.g. reading a baseline as a total). Ambiguous payload keys are renamed to reduce this.
- The dataset's own labels and day counts win over my corrections. "Wed 5 Nov" is actually a Thursday, and `days_to_wedding: 196` is stale by today's date, but the judge checks against the payload. When counts disagree, the message states only the date.
- Templates are deliberately conservative: they score about 35/50, against 39–44 for LLM messages.
- State is in memory, so the service must run as exactly one worker and must not restart during the test.

## What extra context would help most
- Real **available slots** and **service durations** for each merchant, so booking messages aren't limited to seed triggers.
- **Review counts and ratings** in the merchant context. Milestone and review messages currently rely on the trigger payload.
- A clear **"as-of" date** for each trigger. Payload day counts are computed on an unstated date.
- **Offer eligibility** for each offer (new vs returning, weekdays), to avoid pitching new-user offers to lapsed members.
- **Merchant-approved WhatsApp templates**, to fill the `template_name`/`template_params` slots with real IDs.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env            # set LLM_API_KEY (Groq)
uvicorn bot:app --host 0.0.0.0 --port 8080 --workers 1
python run_judge.py all         # local judge (wraps judge_simulator.py without editing it)
python generate_submission.py   # -> submission.jsonl (30 test pairs)
```
