import streamlit as st
from agent.orchestrator import orchestrator
from config.constants import PERSONAS
from utils.helpers import priority_label, confidence_label


def render_chat_tab():
    st.header("Agent Chat")
    st.caption("Type commands like 'Handle this email', 'Draft a friendly reply', 'Summarize only'")

    col1, col2 = st.columns([3, 1])
    with col1:
        email_id = st.text_input("Email ID", placeholder="email_001", key="chat_email_id")
    with col2:
        persona = st.selectbox("Persona", PERSONAS, key="chat_persona")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your command...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if not email_id.strip():
            reply = "⚠️ Please enter an Email ID above before sending a command."
        else:
            with st.spinner("Agent thinking..."):
                try:
                    result = orchestrator.handle_email(
                        email_id.strip(),
                        user_command=user_input,
                        persona=persona,
                        dry_run=False,
                    )
                    if "error" in result:
                        reply = f"❌ {result['error']}"
                    else:
                        plan  = result["plan"]
                        state = result["final_state"]
                        reply = (
                            f"**Planner decision:** {plan.get('explanation', '—')}\n\n"
                            f"**Tools run:** {', '.join(plan.get('tools_to_call', []))}\n\n"
                            f"**Category:** {state.get('category','—')} | "
                            f"**Priority:** {priority_label(state.get('priority'))} | "
                            f"**Confidence:** {confidence_label(state.get('confidence'))}\n\n"
                            f"**Task:** {state.get('task') or '—'}\n\n"
                            f"**Deadline:** {state.get('deadline') or '—'}\n\n"
                            f"**Summary:** {state.get('summary') or '—'}"
                        )
                except Exception as e:
                    reply = f"❌ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()