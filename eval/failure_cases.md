# Failure Cases & Fixes

## Case 1 — Wrong categorization of billing email as "Action Required"

### What happened
Email: "Your invoice INV-2024-089 is due on October 20"
Sender: billing@vendor.com

LLM categorized as: **Action Required** (confidence 0.71)
Correct category: **Billing / Invoice**

### Why it happened
The email contains "due" and a deadline, which triggered the Action Required keywords
in the LLM's reasoning. The billing-specific signals (invoice number, vendor sender)
were underweighted.

### How it was fixed
1. User corrected the category via the Feedback tab
2. `get_feedback_preferences()` stored: `{"sender_category_overrides": {"billing@vendor.com": "Billing / Invoice"}}`
3. On next run, `_apply_feedback_to_state()` in planner.py forced the correct category
4. Heuristic confidence now checks for "invoice" keyword → boosts Billing/Invoice confidence

### Verified fix
Re-ran email_007. Planner log shows:
`Feedback override: sender billing@vendor.com → category forced to 'Billing / Invoice'`

---

## Case 2 — Task extraction failure on travel confirmation email

### What happened
Email: "Your flight to Mumbai is confirmed. PNR: ABC123"
LLM extracted task: "Check in for flight" (confidence 0.61)
Correct result: No task (this is a confirmation, not an action item)

### Why it happened
LLM hallucinated a check-in task from flight confirmation context.

### How it was fixed
1. `SKIP_TASK_CATEGORIES` includes "Travel" — executor skips extract_tasks entirely
2. Planner now enforces this in `_enforce_skip_rules()` — cannot be overridden by LLM
3. Result: task=null for all travel emails, no hallucination possible

---

## Case 3 — Newsletter getting a reply draft generated

### What happened
Early version called generate_reply on a newsletter email.
This wasted an LLM call and produced a nonsense reply to a no-reply address.

### How it was fixed
`SKIP_REPLY_CATEGORIES` includes "Newsletter".
`_enforce_skip_rules()` removes generate_reply from tools_to_call for these categories.
Planner log: `Skipping generate_reply (category=Newsletter)`