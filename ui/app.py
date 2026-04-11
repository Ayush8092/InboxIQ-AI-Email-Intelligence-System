import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from memory.db import init_db
from memory.repository import (
    insert_email, get_email_sources, clear_emails_by_source,
)
from utils.secure_logger import get_secure_logger
from utils.helpers import load_json_file
from config.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    OAUTH_REDIRECT_URI,
)

logger = get_secure_logger("aeoa.app")

st.set_page_config(
    page_title="AEOA — AI Email Agent",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_mock_inbox():
    """Load demo emails — only if not already loaded and user not authenticated."""
    if st.session_state.get("authenticated"):
        return   # never load demo emails when user is logged in

    try:
        emails = load_json_file("data/mock_inbox.json")
        for email in emails:
            email["source"] = "demo"   # explicitly tag as demo
            insert_email(email, source="demo")
        logger.info(f"Demo inbox loaded: {len(emails)} emails")
    except Exception as e:
        logger.warning(f"Mock inbox load failed: {type(e).__name__}")


def handle_oauth_callback():
    """Handle Google OAuth redirect callback."""
    try:
        params = st.query_params
        code   = params.get("code")
        state  = params.get("state", "")
        error  = params.get("error")

        if not code and not error:
            return

        if st.session_state.get("authenticated"):
            st.query_params.clear()
            return

        if error:
            st.session_state["auth_error"] = f"Sign-in cancelled: {error}"
            st.query_params.clear()
            return

        if code:
            from utils.oauth import exchange_code_for_tokens, get_user_info, store_tokens_in_session

            with st.spinner("Completing sign-in..."):
                tokens = exchange_code_for_tokens(
                    code=code,
                    client_id=GOOGLE_CLIENT_ID,
                    client_secret=GOOGLE_CLIENT_SECRET,
                    redirect_uri=OAUTH_REDIRECT_URI,
                )

            if not tokens or "access_token" not in tokens:
                st.session_state["auth_error"] = "Authentication failed. Please try again."
                st.query_params.clear()
                return

            user = get_user_info(tokens["access_token"])
            if not user:
                st.session_state["auth_error"] = "Could not fetch Google profile."
                st.query_params.clear()
                return

            store_tokens_in_session(st.session_state, tokens)
            st.session_state["authenticated"]  = True
            st.session_state["user_name"]      = user.get("name", "User")
            st.session_state["user_email"]     = user.get("email", "")
            st.session_state["user_picture"]   = user.get("picture", "")
            st.session_state["gmail_loaded"]   = False

            logger.info("User authenticated successfully")
            st.query_params.clear()
            st.rerun()

    except Exception as e:
        logger.error(f"OAuth callback error: {type(e).__name__}: {e}")
        st.session_state["auth_error"] = "Unexpected error. Please try again."
        st.query_params.clear()


def render_sidebar():
    with st.sidebar:
        st.title("📧 AEOA")
        st.caption("Autonomous Email Agent")
        st.divider()

        oauth_enabled = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

        if st.session_state.get("authenticated"):
            pic  = st.session_state.get("user_picture","")
            name = st.session_state.get("user_name","User")
            mail = st.session_state.get("user_email","")

            if pic:
                st.image(pic, width=48)
            st.write(f"**{name}**")
            st.caption(mail)
            st.success("✅ Signed in with Google")

            expiry = st.session_state.get("token_expiry", 0)
            mins   = max(0, int((expiry - time.time()) / 60))
            if mins < 5:
                st.warning(f"⚠️ Token expires in {mins} min")
            else:
                st.caption(f"🔑 Token valid ~{mins} min")

            st.divider()

            # ── Email count selector ──────────────────────────────────────
            st.markdown("**📥 Gmail Fetch Settings**")
            email_count = st.slider(
                "Number of emails to fetch",
                min_value=10,
                max_value=75,
                value=st.session_state.get("gmail_fetch_count", 20),
                step=5,
                key="email_count_slider",
                help="How many recent emails to fetch from Gmail",
            )
            st.session_state["gmail_fetch_count"] = email_count

            # ── OCR toggle ────────────────────────────────────────────────
            ocr_enabled = st.checkbox(
                "🔍 Enable OCR (image attachments)",
                value=st.session_state.get("ocr_enabled", False),
                key="ocr_checkbox",
                help="Extract text from image attachments using Google Vision API",
            )
            st.session_state["ocr_enabled"] = ocr_enabled

            if ocr_enabled and not os.getenv("GOOGLE_VISION_API_KEY"):
                st.warning(
                    "⚠️ GOOGLE_VISION_API_KEY not set in .env\n"
                    "OCR will be skipped."
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Load Gmail", use_container_width=True):
                    st.session_state["load_gmail"] = True
            with col2:
                if st.button("🔄 Refresh Token", use_container_width=True):
                    from utils.oauth import get_valid_access_token
                    tok = get_valid_access_token(st.session_state)
                    if tok:
                        st.success("✅ Refreshed")
                    else:
                        st.error("Failed")

            st.divider()

            # ── ML tools ──────────────────────────────────────────────────
            st.markdown("**🤖 ML Model**")
            from memory.repository import get_active_model_version, get_all_feedback
            from services.ml_service import should_auto_retrain

            model_info = get_active_model_version()
            feedback   = get_all_feedback()

            if model_info:
                st.caption(
                    f"v{model_info['version'][:8]} | "
                    f"acc={model_info.get('accuracy',0):.2f}"
                )
            else:
                st.caption("No model trained yet")

            if should_auto_retrain(feedback):
                st.info("💡 New feedback — retrain recommended")

            if st.button("🧪 Simulate Feedback", use_container_width=True):
                from services.feedback_simulator import simulate_feedback
                samples = simulate_feedback(n_samples=20)
                st.success(f"✅ {len(samples)} samples generated")

            if st.button("🤖 Train ML Model", use_container_width=True):
                with st.spinner("Training..."):
                    from services.ml_service import train_model
                    from memory.repository import get_all_emails, get_all_processed
                    source  = "gmail" if st.session_state.get("authenticated") else None
                    emails  = get_all_emails(source=source)
                    proc_map = {p["email_id"]: p for p in get_all_processed(source=source)}
                    result  = train_model(emails, proc_map, feedback)
                    if result.get("success"):
                        st.success("✅ ML model trained!")
                    else:
                        st.info(result.get("reason","Need more data"))

            st.divider()

            if st.button("🔓 Sign Out", use_container_width=True):
                for k in [
                    "authenticated","enc_access_token","enc_refresh_token",
                    "token_expiry","user_name","user_email","user_picture",
                    "gmail_loaded","oauth_state","ml_predictions",
                    "gmail_fetch_count","ocr_enabled",
                ]:
                    st.session_state.pop(k, None)
                st.rerun()

        else:
            st.info("🎯 **Demo Mode**\n20 preloaded sample emails")
            st.divider()

            if oauth_enabled:
                from utils.oauth import get_auth_url
                import secrets as _secrets
                oauth_state = _secrets.token_urlsafe(16)
                st.session_state["oauth_state"] = oauth_state
                auth_url = get_auth_url(
                    client_id=GOOGLE_CLIENT_ID,
                    redirect_uri=OAUTH_REDIRECT_URI,
                    state=oauth_state,
                )
                st.markdown(
                    f"""
                    <a href="{auth_url}" target="_self">
                        <button style="
                            background-color:#4285F4;color:white;
                            border:none;padding:10px 16px;
                            border-radius:4px;font-size:14px;
                            cursor:pointer;width:100%;
                        ">🔐 Sign in with Google</button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Read-only Gmail access. No sensitive data stored.")
            else:
                st.warning("OAuth not configured. Demo mode only.")

            st.divider()
            st.markdown("**Demo Tools**")

            if st.button("🧪 Simulate Feedback", use_container_width=True):
                from services.feedback_simulator import simulate_feedback
                samples = simulate_feedback(n_samples=20)
                st.success(f"✅ {len(samples)} samples generated")

            if st.button("🤖 Train ML Model", use_container_width=True):
                with st.spinner("Training..."):
                    from services.ml_service import train_model
                    from memory.repository import (
                        get_all_emails, get_all_processed, get_all_feedback,
                    )
                    emails   = get_all_emails(source="demo")
                    proc_map = {p["email_id"]: p for p in get_all_processed(source="demo")}
                    feedback = get_all_feedback()
                    result   = train_model(emails, proc_map, feedback)
                    if result.get("success"):
                        st.success("✅ ML model trained!")
                    else:
                        st.info(result.get("reason","Process emails first"))

            if st.session_state.get("auth_error"):
                st.error(st.session_state.pop("auth_error"))

        st.divider()
        st.caption("v2.0 | Groq + scikit-learn")


def load_gmail_emails():
    """Fetch Gmail emails — replaces demo emails, keeps only Gmail."""
    if st.session_state.get("gmail_loaded"):
        return

    from utils.oauth import get_valid_access_token, fetch_gmail_emails

    token = get_valid_access_token(st.session_state)
    if not token:
        st.error("Session expired. Please sign in again.")
        for k in ["authenticated","enc_access_token","enc_refresh_token"]:
            st.session_state.pop(k, None)
        return

    max_results = st.session_state.get("gmail_fetch_count", 20)
    run_ocr     = st.session_state.get("ocr_enabled", False)

    with st.spinner(f"Fetching {max_results} emails from Gmail..."):
        emails = fetch_gmail_emails(
            token,
            max_results=max_results,
            run_ocr=run_ocr,
        )

    if emails:
        # Insert Gmail emails tagged with source='gmail'
        for email in emails:
            insert_email(email, source="gmail")

        st.session_state["gmail_loaded"] = True
        ocr_note = " (with OCR)" if run_ocr else ""
        st.success(f"✅ Loaded {len(emails)} Gmail emails{ocr_note}!")
        st.rerun()
    else:
        st.warning("No emails fetched from Gmail.")


def main():
    init_db()
    handle_oauth_callback()

    if not st.session_state.get("authenticated"):
        load_mock_inbox()

    if st.session_state.get("load_gmail"):
        del st.session_state["load_gmail"]
        load_gmail_emails()

    render_sidebar()

    mode = "Gmail" if st.session_state.get("authenticated") else "Demo"
    st.title("📧 Autonomous Email Operations Agent")
    st.caption(
        f"AI triage · Task extraction · Priority scoring · Reply drafting "
        f"| **{mode} Mode**"
    )

    if not st.session_state.get("authenticated"):
        st.info(
            "👋 **Demo Mode** — 20 preloaded sample emails. "
            "Sign in with Google to process your real inbox."
        )
    else:
        if not st.session_state.get("gmail_loaded"):
            st.info(
                "✅ Signed in! Click **📥 Load Gmail** in the sidebar "
                "to fetch your real emails."
            )

    st.warning(
        "🔒 **Safety:** Never sends emails automatically. "
        "All outputs are read-only drafts."
    )

    # ── Import all tabs ───────────────────────────────────────────────────────
    from ui.inbox_tab        import render_inbox_tab
    from ui.categorized_tab  import render_categorized_tab   # updated version
    from ui.job_task_tab     import render_job_task_tab       # NEW TAB
    from ui.dashboard_tab    import render_dashboard_tab
    from ui.timeline_tab     import render_timeline_tab
    from ui.ml_insights_tab  import render_ml_insights_tab
    from ui.chat_tab         import render_chat_tab
    from ui.drafts_tab       import render_drafts_tab
    from ui.prompts_tab      import render_prompts_tab
    from ui.analytics_tab    import render_analytics_tab

    # ── Render tabs ───────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = st.tabs([
        "📥 Inbox",
       "📂 Categorized",
       "💼 Job Task",       # NEW
       "📊 Dashboard",
       "📅 Timeline",
       "🤖 ML Insights",
       "💬 Chat",
       "✉️ Drafts",
       "🧠 Prompts",
       "📈 Analytics",
       ])


    with t1:  render_inbox_tab()
    with t2:  render_categorized_tab()
    with t3:  render_job_task_tab()      # NEW
    with t4:  render_dashboard_tab()
    with t5:  render_timeline_tab()
    with t6:  render_ml_insights_tab()
    with t7:  render_chat_tab()
    with t8:  render_drafts_tab()
    with t9:  render_prompts_tab()
    with t10: render_analytics_tab()


if __name__ == "__main__":
    main()