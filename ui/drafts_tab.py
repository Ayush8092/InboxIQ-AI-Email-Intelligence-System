import streamlit as st
from memory.repository import get_all_drafts
from memory.db import get_connection
from utils.helpers import truncate


def render_drafts_tab():
    st.header("Reply Drafts")
    st.caption("All generated reply drafts — review before sending manually.")

    drafts = get_all_drafts()

    if not drafts:
        st.info("No drafts yet. Process some emails first.")
        return

    rows = []
    for d in drafts:
        rows.append({
            "Draft ID": str(d["id"]),
            "Email ID": d["email_id"],
            "Subject":  d["subject"],
            "Preview":  truncate(d["body"], 80),
            "Persona":  d["persona"],
            "Created":  d["created_at"],
        })

    st.dataframe(rows, use_container_width=True)

    st.divider()
    st.subheader("View Full Draft")

    draft_id = st.text_input("Enter Draft ID", placeholder="1")

    if st.button("👁 View Draft"):
        if not draft_id.strip():
            st.warning("Enter a draft ID.")
        else:
            conn = get_connection()
            row  = conn.execute(
                "SELECT * FROM drafts WHERE id=?", (draft_id.strip(),)
            ).fetchone()
            conn.close()

            if not row:
                st.error("Draft not found.")
            else:
                st.write(f"**Subject:** {row['subject']}")
                st.text_area("Full Body", value=row["body"], height=250)
                st.write(f"**Persona:** {row['persona']}")
                st.write(f"**Created:** {row['created_at']}")