import streamlit as st
from memory.repository import (
    get_all_emails, get_all_processed, get_all_feedback,
)
from services.ml_service import (
    batch_predict, load_active_model, should_auto_retrain,
    get_monitoring_report, FEATURE_NAMES,
)
from services.online_learning import (
    online_learn, rollback_online_model, reset_online_model,
    get_update_log, get_trust_scores,
)
from services.drift_detector import unified_drift_report
from utils.alerting import get_active_alerts
from utils.helpers import priority_label, truncate


def render_ml_insights_tab():
    st.header("🤖 ML Priority Insights")

    emails_list = get_all_emails()
    processed   = get_all_processed()
    feedback    = get_all_feedback()
    emails_map  = {e["id"]: e for e in emails_list}
    proc_map    = {p["email_id"]: p for p in processed}

    if not processed:
        st.info("Process emails first.")
        return

    # ── Active alerts ──────────────────────────────────────────────────────────
    alerts = get_active_alerts()
    if alerts:
        st.subheader("🚨 Active Alerts")
        for alert in alerts:
            sev = alert.get("severity","info")
            if sev == "critical":
                st.error(f"🔴 **{alert['name']}**: {alert['message']}")
            elif sev == "warning":
                st.warning(f"🟡 **{alert['name']}**: {alert['message']}")
            else:
                st.info(f"🔵 **{alert['name']}**: {alert['message']}")
        st.divider()

    # ── Model status ──────────────────────────────────────────────────────────
    st.subheader("📊 Model Status")
    model_data  = load_active_model()
    update_log  = get_update_log()
    trust_scores = get_trust_scores()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Type",           "LogisticReg" if model_data else "Heuristic")
    c2.metric("Training Samples", model_data.get("n_samples",0) if model_data else 0)
    c3.metric("Online Updates",  len(update_log))
    c4.metric("Tracked Users",  len(trust_scores))
    c5.metric("Feedback Total", len([f for f in feedback if f.get("field")=="priority"]))

    if should_auto_retrain(feedback):
        st.info("💡 New feedback — retrain recommended")

    st.divider()

    # ── Unified drift detection ───────────────────────────────────────────────
    st.subheader("🌊 Unified Drift Detection")

    from memory.repository import get_all_training_data
    from services.ml_service import _prediction_log
    train_data   = get_all_training_data()
    ref_features = [d["features"] for d in train_data[-100:] if d.get("features")]
    ref_labels   = [d["label"] for d in train_data[-100:] if d.get("label")]
    cur_labels   = [p.get("priority",4) for p in processed[-50:] if p.get("priority")]
    cur_features = ref_features[-20:] if len(ref_features) > 20 else ref_features

    if ref_features and len(ref_features) >= 5:
        with st.spinner("Running drift analysis..."):
            drift = unified_drift_report(
                reference_features=ref_features,
                current_features=cur_features,
                feature_names=FEATURE_NAMES,
                predictions=_prediction_log,
                reference_labels=ref_labels,
                current_labels=cur_labels,
            )

        overall = drift.get("overall_severity","none")
        sev_map = {"none":"✅ Stable","moderate":"⚠️ Moderate Drift","high":"🔴 High Drift"}
        st.write(f"**Overall Status:** {sev_map.get(overall,'Unknown')}")

        # Recommendations
        for rec in drift.get("recommendations",[]):
            if "immediate" in rec or "high" in rec.lower():
                st.error(f"🔴 {rec}")
            elif "retrain" in rec or "moderate" in rec.lower():
                st.warning(f"🟡 {rec}")
            else:
                st.success(f"✅ {rec}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Feature Drift (PSI)**")
            fd = drift.get("feature_drift",{})
            st.write(f"Detected: {'✅ Yes' if fd.get('drift_detected') else '✅ No'}")
            st.write(f"High drift features: {fd.get('n_high_drift',0)}")
            if fd.get("high_drift_features"):
                st.write(", ".join(fd["high_drift_features"]))

        with col2:
            st.markdown("**Concept Drift (Accuracy)**")
            cd = drift.get("concept_drift",{})
            st.write(f"Detected: {'⚠️ Yes' if cd.get('drift_detected') else '✅ No'}")
            if cd.get("accuracy_drop") is not None:
                st.write(f"Accuracy drop: {cd.get('accuracy_drop',0):.1%}")
            st.write(f"Window: {cd.get('window_size',0)} samples")

        with col3:
            st.markdown("**Label Drift (Distribution)**")
            ld = drift.get("label_drift",{})
            st.write(f"Detected: {'⚠️ Yes' if ld.get('drift_detected') else '✅ No'}")
            st.write(f"KL Divergence: {ld.get('kl_divergence',0):.4f}")
    else:
        st.info("Insufficient training data for drift analysis. Train the model first.")

    st.divider()

    # ── Online learning controls ──────────────────────────────────────────────
    st.subheader("⚡ Online Learning")

    col1, col2, col3 = st.columns(3)
    with col1:
        ol_email = st.selectbox(
            "Email", [e["id"] for e in emails_list], key="ol_email"
        )
    with col2:
        ol_priority = st.selectbox(
            "Correct Priority", [1,2,3,4,5,6,7],
            format_func=priority_label,
            key="ol_priority"
        )
    with col3:
        ol_user = st.text_input("User ID", value="demo_user", key="ol_user")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("⚡ Apply Update", type="primary"):
            email = emails_map.get(ol_email,{})
            proc  = proc_map.get(ol_email,{})
            result = online_learn(email, proc, ol_priority, user_id=ol_user)
            if result.get("success"):
                st.success(
                    f"✅ Model updated | "
                    f"Trust: {result.get('trust_score',0):.2f} | "
                    f"Updates: {result.get('n_updates',0)}"
                )
                from memory.repository import insert_feedback
                insert_feedback(ol_email,"priority",str(proc.get("priority","")),str(ol_priority))
            else:
                st.warning(f"⚠️ {result.get('reason','Update rejected')}")

    with col_b:
        if st.button("↩️ Rollback 1 Step"):
            ok = rollback_online_model(steps=1)
            st.success("✅ Rolled back") if ok else st.error("❌ No snapshots")

    with col_c:
        if st.button("🗑 Reset Online Model"):
            ok = reset_online_model()
            st.success("✅ Reset") if ok else st.error("❌ Failed")

    st.divider()

    # ── User trust scores ─────────────────────────────────────────────────────
    st.subheader("👤 User Trust Scores")
    if trust_scores:
        rows = [
            {
                "User ID":    uid,
                "Trust Score": f"{score:.2f}",
                "Status":     "✅ Trusted" if score >= 0.6 else "⚠️ Low Trust",
            }
            for uid, score in sorted(trust_scores.items(), key=lambda x: -x[1])
        ]
        st.dataframe(rows, use_container_width=True)
        st.caption(
            "Trust scores updated automatically based on feedback quality. "
            "Users below 0.40 have feedback ignored to protect model quality."
        )
    else:
        st.info("No user trust scores yet.")

    st.divider()

    # ── Online update log ─────────────────────────────────────────────────────
    st.subheader("📋 Online Update Log")
    if update_log:
        import datetime
        rows = [
            {
                "Email":    u.get("email_id","")[:20],
                "Priority": priority_label(u.get("priority",4)),
                "User":     u.get("user_id",""),
                "Trust":    f"{u.get('trust_score',0):.2f}",
                "Time":     datetime.datetime.utcfromtimestamp(
                    u.get("ts",0)
                ).strftime("%H:%M:%S"),
            }
            for u in reversed(update_log[-20:])
        ]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No online updates yet.")

    st.divider()

    # ── Model history ─────────────────────────────────────────────────────────
    st.subheader("📚 Model History")
    from memory.repository import get_model_history
    history = get_model_history()
    if history:
        rows = [
            {
                "Version":  h.get("version","")[:14],
                "Accuracy": f"{h.get('accuracy',0):.3f}",
                "Samples":  h.get("n_samples",0),
                "Active":   "✅" if h.get("is_active") else "—",
                "Trained":  h.get("trained_at","")[:16],
            }
            for h in history
        ]
        st.dataframe(rows, use_container_width=True)