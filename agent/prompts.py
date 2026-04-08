from config.constants import CATEGORIES

_cats_csv = ", ".join(f'"{c}"' for c in CATEGORIES)

DEFAULT_PROMPTS = {

"categorize": (
    'Classify this email into one category. Reply with ONLY JSON.\n\n'
    'Subject: {subject}\n'
    'Body: {body}\n\n'
    f'Categories: {_cats_csv}\n\n'
    'Important rules:\n'
    '- Shipping notifications, order updates → Social / Notification\n'
    '- Flash sales, discounts, promotions → Social / Notification\n'
    '- Password resets, account alerts → Social / Notification\n'
    '- Server down, outages, 503 errors → Alert / Urgent\n'
    '- Meeting invites, lunch plans → Meeting / Event\n\n'
    'Reply format: {{"category": "CATEGORY_NAME", "confidence": 0.85}}'
),

"extract_tasks": (
    'You are an intelligent email assistant. Extract and enrich actionable insights.\n\n'
    'Email Subject: {subject}\n'
    'Email Body: {body}\n\n'
    'Perform the following:\n\n'
    '1. CONTEXT-AWARE TASK ENRICHMENT\n'
    '   - Extract the main task.\n'
    '   - Enrich with full context (event, purpose, deadline, entities).\n'
    '   BAD: "Prepare self-assessment"\n'
    '   GOOD: "Prepare self-assessment before the Oct 22 performance review meeting with manager"\n\n'
    '2. MULTI-STEP TASK GENERATION\n'
    '   - If the email implies a workflow, break into 2-5 ordered steps.\n'
    '   - Example for "Server is down":\n'
    '     steps: ["Check server logs for errors", "Restart the affected service",\n'
    '             "Verify system health", "Notify team about the incident"]\n\n'
    '3. SMART NO-TASK HANDLING\n'
    '   - If no direct action required, classify as: reminder, calendar_event, or informational\n'
    '   - Newsletter → informational: "Read later if relevant"\n'
    '   - Webinar confirmation → calendar_event: "Attend AI webinar on [date]"\n'
    '   - Flight confirmation → reminder: "Keep ticket ready before travel"\n\n'
    '4. DEADLINE EXTRACTION\n'
    '   - Extract explicit deadline if present.\n'
    '   - If not explicit, infer from context (meeting date → task before meeting).\n\n'
    'Output ONLY this JSON:\n'
    '{{\n'
    '  "type": "task|multi_step|reminder|calendar_event|informational",\n'
    '  "task": "Fully enriched single-line task (never vague)",\n'
    '  "steps": ["step1", "step2"] or null,\n'
    '  "deadline": "YYYY-MM-DD or null",\n'
    '  "confidence": 0.85\n'
    '}}'
),

"summarize": (
    'Summarize this email in 1-2 sentences.\n\n'
    'Subject: {subject}\n'
    'Sender: {sender}\n'
    'Body: {body}\n\n'
    'Write the summary only, no labels or preamble.'
),

"generate_reply": (
    'Write an email reply using the {persona} persona.\n'
    'Persona: {persona_description}\n\n'
    'Original email:\n'
    'Subject: {subject}\n'
    'From: {sender}\n'
    'Body: {body}\n\n'
    'Reply with ONLY JSON:\n'
    '{{"subject": "Re: {subject}", "body": "REPLY_BODY", "follow_ups": ["ACTION1"]}}\n\n'
    'Rules:\n'
    '- Draft only — never mention sending\n'
    '- Under 150 words\n'
    '- Use \\\\n for line breaks inside body — never actual newlines'
),

"planner": (
    'You are an email operations planner. Decide which tools to run.\n\n'
    'Email: {email_id}\n'
    'Subject: {subject}\n'
    'Sender: {sender}\n'
    'Current state: {current_state}\n'
    'User preferences: {user_preferences}\n'
    'Command: {user_command}\n\n'
    'Tools:\n'
    '- categorize_email: classify email type\n'
    '- extract_tasks: find tasks, steps, and deadlines\n'
    '- compute_priority: score urgency 1-7\n'
    '- summarize_email: generate short summary\n'
    '- generate_reply: draft a reply\n'
    '- snooze_email: defer this email\n\n'
    'Reply with ONLY JSON:\n'
    '{{"tools_to_call": ["tool1", "tool2"], '
    '"skip_reasons": {{"tool": "reason"}}, '
    '"needs_review": false, '
    '"explanation": "reasoning"}}'
),
}