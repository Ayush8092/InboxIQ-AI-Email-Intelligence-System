import streamlit as st
from datetime import date, datetime
from collections import Counter
from memory.repository import (
    get_metrics_summary, get_all_processed,
    get_all_emails, get_total_llm_calls, get_all_feedback,
)
from utils.helpers import priority_label
from utils.cache import cache_stats
from config.constants import CATEGORIES, PRIORITY_LABELS


def render_analytics_tab():
    st.header("Analytics Dashboard")

    col_r, col_c = st.columns([1, 1])
    with col_r:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_c:
        if st.button("🗑 Clear Cache"):
            from utils.cache import clear_cache
            clear_cache()
            st.success("Cache cleared.")

    emails    = get_all_emails()
    processed = get_all_processed()
    metrics   = get_metrics_summary()
    feedback  = get_all_feedback()
    needs_rev = sum(1 for p in processed if p.get("needs_review"))

    # ── Top metrics ───────────────────────────────────────────────────────────
    st.subheader("Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📧 Emails",     len(emails))
    c2.metric("✅ Processed",  len(processed))
    c3.metric("🤖 LLM Calls",  get_total_llm_calls())
    c4.metric("⚠️ Review",     needs_rev)
    c5.metric("✏️ Corrections", len(feedback))
    auto = len(processed) - needs_rev
    c6.metric("🚀 Auto-handled", auto)

    # Auto-handle rate
    if processed:
        auto_rate = auto / len(processed) * 100
        color = "normal" if auto_rate >= 75 else "inverse"
        st.progress(int(auto_rate), text=f"Auto-handle rate: {auto_rate:.0f}% (target: 75-90%)")

    st.divider()

    # ── Cache performance ─────────────────────────────────────────────────────
    st.subheader("⚡ Cache Performance")
    cs   = cache_stats()
    ca, cb, cc, cd = st.columns(4)
    ca.metric("Cache Hits",   cs["hits"])
    cb.metric("Cache Misses", cs["misses"])
    cc.metric("Hit Rate",     f"{cs['hit_rate']}%")
    cd.metric("Cache Size",   cs["cache_size"])

    st.divider()

    # ── Category + Priority distribution ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Category Distribution")
        cat_counts = Counter({c: 0 for c in CATEGORIES})
        for p in processed:
            c = p.get("category")
            if c in cat_counts:
                cat_counts[c] += 1
        cat_data = {k: v for k, v in cat_counts.items() if v > 0}
        if cat_data:
            st.bar_chart(cat_data)
            total = len(processed)
            for cat, count in sorted(cat_data.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total else 0
                st.write(f"**{cat}:** {count} ({pct:.0f}%)")
        else:
            st.info("No data yet.")

    with col2:
        st.subheader("🎯 Priority Distribution")
        prio_counts = Counter()
        for p in processed:
            pr = p.get("priority")
            if pr:
                prio_counts[priority_label(pr)] += 1

        if prio_counts:
            st.bar_chart(dict(prio_counts))

            # Show balance — warn if too many Minimal
            minimal = sum(v for k, v in prio_counts.items() if "Minimal" in k or "Negligible" in k)
            critical = sum(v for k, v in prio_counts.items() if "Critical" in k or "High" in k)
            if processed:
                if minimal / len(processed) > 0.50:
                    st.warning("⚠️ Over 50% emails are Minimal priority — check priority rules")
                if critical / len(processed) > 0.30:
                    st.info("ℹ️ High proportion of critical emails detected")
        else:
            st.info("No data yet.")

    st.divider()

    # ── Confidence distribution ───────────────────────────────────────────────
    st.subheader("📊 Confidence Distribution")
    confs = [p.get("confidence") for p in processed if p.get("confidence") is not None]
    if confs:
        import numpy as np
        avg  = sum(confs) / len(confs)
        high = sum(1 for c in confs if c >= 0.80)
        med  = sum(1 for c in confs if 0.60 <= c < 0.80)
        low  = sum(1 for c in confs if c < 0.60)
        std  = round(float(np.std(confs)), 3) if confs else 0

        ca, cb, cc, cd, ce = st.columns(5)
        ca.metric("Avg Conf",    f"{avg:.0%}")
        cb.metric("Std Dev",     std, help="Higher = more spread (better calibration)")
        cc.metric("✅ High≥80%",  high)
        cd.metric("⚠️ Med 60-80%", med)
        ce.metric("❌ Low<60%",  low)

        # Confidence distribution histogram
        buckets = {"<50%":0, "50-60%":0, "60-70%":0, "70-80%":0, "80-88%":0}
        for c in confs:
            if c < 0.50:   buckets["<50%"] += 1
            elif c < 0.60: buckets["50-60%"] += 1
            elif c < 0.70: buckets["60-70%"] += 1
            elif c < 0.80: buckets["70-80%"] += 1
            else:          buckets["80-88%"] += 1
        st.bar_chart(buckets)
    else:
        st.info("No confidence data yet.")

    st.divider()

    # ── Deadline summary ──────────────────────────────────────────────────────
    st.subheader("📅 Task Deadlines")
    today    = date.today()
    overdue  = due_today = due_week = no_dl = 0
    for p in processed:
        dl = p.get("deadline")
        if not dl:
            no_dl += 1
            continue
        try:
            d = datetime.fromisoformat(str(dl)).date()
            if d < today:              overdue   += 1
            elif d == today:           due_today += 1
            elif (d - today).days <= 7: due_week  += 1
        except Exception:
            no_dl += 1

    ca, cb, cc, cd = st.columns(4)
    ca.metric("🔴 Overdue",       overdue)
    cb.metric("🟡 Due Today",     due_today)
    cc.metric("🟠 Due This Week", due_week)
    cd.metric("⚪ No Deadline",   no_dl)

    st.divider()

    # ── Tool performance ──────────────────────────────────────────────────────
    st.subheader("⚡ Tool Performance")
    if metrics:
        rows = []
        for tool, m in metrics.items():
            calls = m.get("calls", 0)
            err   = m.get("errors", 0)
            rows.append({
                "Tool":        tool,
                "Calls":       calls,
                "Avg Latency": f"{m.get('avg_latency_ms', 0):.0f} ms",
                "Errors":      err,
                "Error Rate":  f"{err/max(calls,1)*100:.0f}%",
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No metrics yet.")

    st.divider()

    # ── Needs review breakdown ────────────────────────────────────────────────
    st.subheader("⚠️ Needs Review")
    review_list = [p for p in processed if p.get("needs_review")]
    if review_list:
        rows = []
        for p in review_list:
            rows.append({
                "Email ID":      p["email_id"],
                "Category":      p.get("category") or "—",
                "Confidence":    f"{p.get('confidence',0):.0%}" if p.get("confidence") else "—",
                "Review Reason": p.get("review_reason") or "—",
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.success("✅ No emails need review.")

    st.divider()

    # ── Evaluation results ────────────────────────────────────────────────────
    st.subheader("🧪 Evaluation Results")
    import json, os
    if os.path.exists("eval/results.json"):
        with open("eval/results.json") as f:
            eval_data = json.load(f)
        m = eval_data.get("metrics", {})
        ca, cb, cc = st.columns(3)
        ca.metric("Cat Accuracy",  f"{m.get('cat_accuracy',0)}%")
        cb.metric("Task Accuracy", f"{m.get('task_accuracy',0)}%")
        cc.metric("Macro F1",      m.get("macro_f1", 0))

        per_cat = m.get("per_category", {})
        if per_cat:
            rows = [
                {"Category": cat, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"]}
                for cat, v in per_cat.items()
            ]
            st.dataframe(rows, use_container_width=True)
    else:
        st.info("Run `python -m eval.run_eval` to generate results.")