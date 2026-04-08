import streamlit as st
from memory.repository import get_prompt, set_prompt
from agent.prompts import DEFAULT_PROMPTS

PROMPT_NAMES = ["categorize", "extract_tasks", "summarize", "generate_reply", "planner"]


def render_prompts_tab():
    st.header("Prompt Brain & Personas")
    st.caption("Edit the LLM prompts used by each tool. Changes are saved to DB and used immediately.")

    prompt_name = st.selectbox("Select Prompt to Edit", PROMPT_NAMES)

    saved   = get_prompt(prompt_name)
    current = saved if saved else DEFAULT_PROMPTS.get(prompt_name, "")

    template = st.text_area("Prompt Template", value=current, height=400, key=f"prompt_{prompt_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Prompt", type="primary"):
            set_prompt(prompt_name, template.strip())
            st.success(f"✅ Prompt '{prompt_name}' saved.")
    with col2:
        if st.button("🔄 Reset to Default"):
            default = DEFAULT_PROMPTS.get(prompt_name, "")
            set_prompt(prompt_name, default)
            st.success("🔄 Reset to default.")
            st.rerun()
    with col3:
        if st.button("📂 Reload from DB"):
            st.rerun()

    st.divider()
    st.subheader("Personas")
    st.markdown("""
| Persona | Tone Description |
|---------|-----------------|
| **Formal** | Professional, full sentences, no contractions |
| **Friendly** | Warm, conversational, approachable |
| **Concise** | Direct, minimal words, no filler phrases |
    """)