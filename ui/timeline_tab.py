"""
Task Timeline — calendar view of extracted tasks with deadlines,
priorities, and multi-step breakdowns.
"""
import streamlit as st
from datetime import date, datetime, timedelta
from collections import defaultdict
from memory.repository import get_all_processed, get_all_emails
from utils.helpers import priority_label, truncate

_PRIORITY_COLORS = {
    1: "#FF4444",   # Critical — red
    2: "#FF8C00",   # High — orange
    3: "#FFD700",   # Medium — yellow
    4: "#4169E1",   # Low — blue
    5: "#808080",   # Very Low — gray
    6: "#A9A9A9",
    7: "#C0C0C0",
}

_TYPE_ICONS = {
    "task":          "✅",
    "multi_step":    "📋",
    "reminder":      "🔔",
    "calendar_event":"📅",
    "informational": "ℹ️",
}


def _week_range(ref_date: date, week_offset: int = 0) -> tuple[date, date]:
    """Get start and end of a week relative to ref_date."""
    start = ref_date - timedelta(days=ref_date.weekday()) + timedelta(weeks=week_offset)
    end   = start + timedelta(days=6)
    return start, end


def _render_task_card(p: dict, email: dict, compact: bool = False):
    """Render a single task card."""
    task      = p.get("task","No task")
    priority  = p.get("priority", 5)
    t_type    = p.get("task_type","task")
    steps     = p.get("steps") or []
    deadline  = p.get("deadline","")
    icon      = _TYPE_ICONS.get(t_type,"✅")
    p_label   = priority_label(priority)
    color     = _PRIORITY_COLORS.get(priority,"#808080")
    subject   = email.get("subject","")
    sender    = email.get("sender","")

    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {color};
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 0 8px 8px 0;
            background: rgba(0,0,0,0.03);
        ">
            <div style="font-weight:600; font-size:14px;">
                {icon} {truncate(task, 60)}
            </div>
            <div style="font-size:12px; color:#666; margin-top:2px;">
                {p_label} | {truncate(subject,40)}
            </div>
            <div style="font-size:11px; color:#888;">
                From: {sender}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not compact and steps:
        with st.expander(f"📋 {len(steps)} steps"):
            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}")


def render_timeline_tab():
    st.header("📅 Task Timeline")

    processed   = get_all_processed()
    emails_list = get_all_emails()
    emails_map  = {e["id"]: e for e in emails_list}

    # Filter to tasks with deadlines
    tasks_with_dl  = [p for p in processed if p.get("deadline") and p.get("task")]
    tasks_no_dl    = [p for p in processed if not p.get("deadline") and p.get("task")]
    today          = date.today()

    # ── View selector ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        view = st.radio(
            "View", ["Week", "Month", "All Tasks", "By Priority"],
            horizontal=True, key="timeline_view"
        )
    with col2:
        if view == "Week":
            week_offset = st.number_input(
                "Week offset", min_value=-4, max_value=8,
                value=0, key="week_offset"
            )
        else:
            week_offset = 0
    with col3:
        filter_priority = st.selectbox(
            "Priority filter",
            ["All", "Critical (1)", "High (2)", "Medium (3)", "Low (4+)"],
            key="priority_filter"
        )

    # Priority filter
    def passes_priority(p: dict) -> bool:
        pr = p.get("priority", 5)
        if filter_priority == "All":              return True
        if filter_priority == "Critical (1)":     return pr == 1
        if filter_priority == "High (2)":         return pr == 2
        if filter_priority == "Medium (3)":       return pr == 3
        if filter_priority == "Low (4+)":         return pr >= 4
        return True

    # ── Stats bar ─────────────────────────────────────────────────────────────
    overdue = sum(
        1 for p in tasks_with_dl
        if datetime.strptime(p["deadline"], "%Y-%m-%d").date() < today
    )
    due_today = sum(
        1 for p in tasks_with_dl
        if datetime.strptime(p["deadline"], "%Y-%m-%d").date() == today
    )
    due_week  = sum(
        1 for p in tasks_with_dl
        if 0 < (datetime.strptime(p["deadline"], "%Y-%m-%d").date() - today).days <= 7
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Overdue",       overdue)
    c2.metric("🟡 Due Today",     due_today)
    c3.metric("🟠 Due This Week", due_week)
    c4.metric("📋 Total Tasks",   len(tasks_with_dl) + len(tasks_no_dl))

    st.divider()

    # ── Week view ─────────────────────────────────────────────────────────────
    if view == "Week":
        start, end = _week_range(today, week_offset)
        st.subheader(
            f"Week of {start.strftime('%b %d')} — {end.strftime('%b %d, %Y')}"
        )

        # Build day buckets
        day_buckets = defaultdict(list)
        for p in tasks_with_dl:
            if not passes_priority(p):
                continue
            try:
                dl = datetime.strptime(p["deadline"], "%Y-%m-%d").date()
                if start <= dl <= end:
                    day_buckets[dl].append(p)
            except Exception:
                pass

        # Render 7 columns for the week
        day_cols = st.columns(7)
        for i, col in enumerate(day_cols):
            day        = start + timedelta(days=i)
            day_tasks  = day_buckets.get(day, [])
            is_today   = day == today
            is_weekend = day.weekday() >= 5

            with col:
                header_style = "🔵" if is_today else ("⚫" if is_weekend else "⚪")
                st.markdown(
                    f"**{header_style} {day.strftime('%a')}**\n\n"
                    f"_{day.strftime('%b %d')}_"
                )
                if day_tasks:
                    for p in sorted(day_tasks, key=lambda x: x.get("priority",7)):
                        email = emails_map.get(p["email_id"],{})
                        task  = p.get("task","")
                        pr    = p.get("priority",5)
                        color = _PRIORITY_COLORS.get(pr,"#808080")
                        icon  = _TYPE_ICONS.get(p.get("task_type","task"),"✅")
                        st.markdown(
                            f"""<div style="
                                border-left:3px solid {color};
                                padding:4px 6px;
                                margin:2px 0;
                                font-size:11px;
                                border-radius:0 4px 4px 0;
                                background:rgba(0,0,0,0.04);
                            ">{icon} {truncate(task,30)}</div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("—")

        st.divider()

        # Overdue section for week view
        overdue_tasks = [
            p for p in tasks_with_dl
            if passes_priority(p)
            and datetime.strptime(p["deadline"], "%Y-%m-%d").date() < today
        ]
        if overdue_tasks:
            st.subheader("🔴 Overdue Tasks")
            for p in sorted(overdue_tasks, key=lambda x: x.get("priority",7)):
                email = emails_map.get(p["email_id"],{})
                dl    = datetime.strptime(p["deadline"], "%Y-%m-%d").date()
                days  = (today - dl).days
                col1, col2 = st.columns([4, 1])
                with col1:
                    _render_task_card(p, email)
                with col2:
                    st.error(f"{days}d overdue")

    # ── Month view ────────────────────────────────────────────────────────────
    elif view == "Month":
        st.subheader(f"Month: {today.strftime('%B %Y')}")

        month_start = today.replace(day=1)
        # Find next month start
        if today.month == 12:
            month_end = today.replace(year=today.year+1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = today.replace(month=today.month+1, day=1) - timedelta(days=1)

        # Group by week
        week_groups = defaultdict(list)
        for p in tasks_with_dl:
            if not passes_priority(p):
                continue
            try:
                dl = datetime.strptime(p["deadline"], "%Y-%m-%d").date()
                if month_start <= dl <= month_end:
                    week_num = (dl - month_start).days // 7
                    week_groups[week_num].append(p)
            except Exception:
                pass

        for week_num in sorted(week_groups.keys()):
            week_start = month_start + timedelta(weeks=week_num)
            week_end   = min(week_start + timedelta(days=6), month_end)
            tasks      = week_groups[week_num]

            with st.expander(
                f"Week {week_num+1}: {week_start.strftime('%b %d')} — "
                f"{week_end.strftime('%b %d')} ({len(tasks)} tasks)"
            ):
                for p in sorted(tasks, key=lambda x: (x.get("deadline",""), x.get("priority",7))):
                    email = emails_map.get(p["email_id"],{})
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        _render_task_card(p, email)
                    with col2:
                        dl_date = datetime.strptime(p["deadline"], "%Y-%m-%d").date()
                        days    = (dl_date - today).days
                        if days < 0:
                            st.error(f"Overdue {abs(days)}d")
                        elif days == 0:
                            st.warning("Today!")
                        else:
                            st.info(f"In {days}d")

    # ── All tasks list ─────────────────────────────────────────────────────────
    elif view == "All Tasks":
        st.subheader("All Tasks with Deadlines")

        sorted_tasks = sorted(
            [p for p in tasks_with_dl if passes_priority(p)],
            key=lambda x: (x.get("deadline","9999"), x.get("priority",7))
        )

        if sorted_tasks:
            prev_month = None
            for p in sorted_tasks:
                email   = emails_map.get(p["email_id"],{})
                dl_str  = p.get("deadline","")
                try:
                    dl_date   = datetime.strptime(dl_str, "%Y-%m-%d").date()
                    month_str = dl_date.strftime("%B %Y")
                    is_past   = dl_date < today
                except Exception:
                    month_str = "Unknown"
                    is_past   = False

                if month_str != prev_month:
                    st.markdown(f"### 📆 {month_str}")
                    prev_month = month_str

                col1, col2, col3 = st.columns([5, 1, 1])
                with col1:
                    _render_task_card(p, email, compact=False)
                with col2:
                    st.write(priority_label(p.get("priority",5)))
                with col3:
                    if is_past:
                        st.error("Overdue")
                    elif dl_date == today:
                        st.warning("Today")
                    else:
                        days = (dl_date - today).days
                        st.info(f"{days}d left")
        else:
            st.info("No tasks with deadlines found.")

        # Tasks without deadlines
        no_dl_filtered = [p for p in tasks_no_dl if passes_priority(p)]
        if no_dl_filtered:
            st.divider()
            st.subheader("📌 Tasks Without Deadlines")
            for p in sorted(no_dl_filtered, key=lambda x: x.get("priority",7)):
                email = emails_map.get(p["email_id"],{})
                _render_task_card(p, email, compact=True)

    # ── By Priority ───────────────────────────────────────────────────────────
    elif view == "By Priority":
        st.subheader("Tasks Grouped by Priority")

        all_tasks = [p for p in processed if p.get("task")]
        priority_groups = defaultdict(list)
        for p in all_tasks:
            if passes_priority(p):
                priority_groups[p.get("priority",5)].append(p)

        for pr in sorted(priority_groups.keys()):
            tasks_in_group = priority_groups[pr]
            p_label        = priority_label(pr)
            color          = _PRIORITY_COLORS.get(pr,"#808080")

            st.markdown(
                f"<h4 style='color:{color}'>{p_label} — {len(tasks_in_group)} tasks</h4>",
                unsafe_allow_html=True,
            )

            for p in sorted(
                tasks_in_group,
                key=lambda x: (x.get("deadline") or "9999", x.get("email_id",""))
            ):
                email = emails_map.get(p["email_id"],{})
                col1, col2 = st.columns([5, 1])
                with col1:
                    _render_task_card(p, email, compact=True)
                with col2:
                    dl = p.get("deadline","")
                    if dl:
                        try:
                            dl_date = datetime.strptime(dl, "%Y-%m-%d").date()
                            days    = (dl_date - today).days
                            if days < 0:
                                st.error(f"Overdue")
                            elif days == 0:
                                st.warning("Today")
                            else:
                                st.caption(f"📅 {dl}")
                        except Exception:
                            st.caption(f"📅 {dl}")
                    else:
                        st.caption("No deadline")

            st.divider()