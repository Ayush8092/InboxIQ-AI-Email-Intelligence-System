"""
Full evaluation pipeline.
Run: python -m eval.run_eval
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import init_db
from memory.repository import insert_email, get_processed
from agent.orchestrator import orchestrator
from utils.logger import setup_logger

logger = setup_logger("eval")

LABELED_DATA = [
    {"id":"eval_001","subject":"Q3 Budget Report - Action Required by Friday",
     "body":"Please review the Q3 budget report and send approval by Friday EOD. This is urgent.",
     "sender":"boss@company.com","timestamp":"2024-10-14T09:15:00",
     "expected_category":"Action Required","expected_task_not_null":True},

    {"id":"eval_002","subject":"Your Amazon order has shipped!",
     "body":"Your order #112-3456789 has shipped and will arrive by Thursday.",
     "sender":"shipping@amazon.com","timestamp":"2024-10-14T10:02:00",
     "expected_category":"Social / Notification","expected_task_not_null":False},

    {"id":"eval_003","subject":"Weekly Newsletter - Tech Digest #45",
     "body":"This week in tech: OpenAI announces new model. Unsubscribe here.",
     "sender":"newsletter@techdigest.io","timestamp":"2024-10-14T08:00:00",
     "expected_category":"Newsletter","expected_task_not_null":False},

    {"id":"eval_004","subject":"Server is DOWN - Immediate attention needed",
     "body":"URGENT: Production server is not responding. Users getting 503 errors immediately.",
     "sender":"alerts@monitoring.com","timestamp":"2024-10-14T13:45:00",
     "expected_category":"Alert / Urgent","expected_task_not_null":True},

    {"id":"eval_005","subject":"Your invoice INV-2024-089 is due",
     "body":"Invoice INV-2024-089 for $2,340 is due on October 20, 2024. Please arrange payment.",
     "sender":"billing@vendor.com","timestamp":"2024-10-13T15:00:00",
     "expected_category":"Billing / Invoice","expected_task_not_null":True},

    {"id":"eval_006","subject":"Interview invite: Senior Engineer role",
     "body":"We would like to invite you for a technical interview for the Senior Engineer position next week.",
     "sender":"recruiting@techcorp.com","timestamp":"2024-10-14T13:00:00",
     "expected_category":"Job / Recruitment","expected_task_not_null":True},

    {"id":"eval_007","subject":"Your flight to Mumbai is confirmed",
     "body":"Booking confirmed: IndiGo 6E-204, Delhi to Mumbai, Oct 19 at 07:20. PNR: ABC123.",
     "sender":"noreply@indigo.com","timestamp":"2024-10-13T14:00:00",
     "expected_category":"Travel","expected_task_not_null":False},

    {"id":"eval_008","subject":"Team lunch this Friday at 12:30",
     "body":"We're doing a team lunch this Friday at Nando's at 12:30pm. Please confirm attendance.",
     "sender":"rachel@company.com","timestamp":"2024-10-14T09:45:00",
     "expected_category":"Meeting / Event","expected_task_not_null":True},

    {"id":"eval_009","subject":"Performance review scheduled for Oct 22",
     "body":"Your annual performance review has been scheduled for October 22 at 2:00 PM. Please prepare a self-assessment form.",
     "sender":"hr@company.com","timestamp":"2024-10-12T10:00:00",
     "expected_category":"Meeting / Event","expected_task_not_null":True},

    {"id":"eval_010","subject":"Critical security patch - deploy by EOD",
     "body":"A critical CVE has been identified. All environments must be patched by end of day today. Please confirm once done.",
     "sender":"cto@company.com","timestamp":"2024-10-14T12:00:00",
     "expected_category":"Alert / Urgent","expected_task_not_null":True},
]


def compute_metrics(results: list[dict]) -> dict:
    total        = len(results)
    cat_correct  = sum(1 for r in results if r["cat_ok"])
    task_correct = sum(1 for r in results if r["task_ok"])

    categories = list({r["expected_cat"] for r in results})
    per_cat    = {}
    for cat in categories:
        tp = sum(1 for r in results if r["expected_cat"] == cat and r["got_cat"] == cat)
        fp = sum(1 for r in results if r["expected_cat"] != cat and r["got_cat"] == cat)
        fn = sum(1 for r in results if r["expected_cat"] == cat and r["got_cat"] != cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
        per_cat[cat] = {
            "precision": round(precision, 2),
            "recall":    round(recall, 2),
            "f1":        round(f1, 2),
            "tp": tp, "fp": fp, "fn": fn,
        }

    macro_f1    = round(sum(v["f1"] for v in per_cat.values()) / len(per_cat), 3)
    macro_prec  = round(sum(v["precision"] for v in per_cat.values()) / len(per_cat), 3)
    macro_rec   = round(sum(v["recall"]    for v in per_cat.values()) / len(per_cat), 3)

    return {
        "total":          total,
        "cat_correct":    cat_correct,
        "task_correct":   task_correct,
        "cat_accuracy":   round(cat_correct  / total * 100, 1),
        "task_accuracy":  round(task_correct / total * 100, 1),
        "macro_f1":       macro_f1,
        "macro_precision": macro_prec,
        "macro_recall":   macro_rec,
        "per_category":   per_cat,
    }


def run_evaluation():
    init_db()
    for item in LABELED_DATA:
        email = {k: v for k, v in item.items()
                 if k not in ("expected_category","expected_task_not_null")}
        insert_email(email)

    print("\n" + "="*60)
    print("  AEOA Evaluation Pipeline")
    print("="*60 + "\n")

    results = []
    for item in LABELED_DATA:
        orchestrator.handle_email(item["id"], user_command="Handle this email")
        state = get_processed(item["id"]) or {}

        got_cat      = state.get("category", "") or ""
        got_task     = state.get("task")
        got_conf     = state.get("confidence") or 0
        got_nr       = bool(state.get("needs_review"))
        got_reason   = state.get("review_reason") or ""
        exp_cat      = item["expected_category"]
        exp_task_val = item["expected_task_not_null"]

        cat_ok  = got_cat == exp_cat
        task_ok = (got_task is not None) == exp_task_val
        status  = "✅" if (cat_ok and task_ok) else "❌"

        print(
            f"{status} {item['id']:<12} | "
            f"cat: '{exp_cat}' → '{got_cat}' {'✓' if cat_ok else '✗'} | "
            f"task: {exp_task_val} → {bool(got_task)} {'✓' if task_ok else '✗'} | "
            f"conf: {got_conf:.2f} | "
            f"review: {'Yes' if got_nr else 'No'}"
        )
        if got_reason:
            print(f"             reason: {got_reason}")

        results.append({
            "id":            item["id"],
            "subject":       item["subject"][:40],
            "expected_cat":  exp_cat,
            "got_cat":       got_cat,
            "cat_ok":        cat_ok,
            "expected_task": exp_task_val,
            "got_task":      bool(got_task),
            "task_ok":       task_ok,
            "confidence":    got_conf,
            "needs_review":  got_nr,
            "review_reason": got_reason,
        })

    metrics = compute_metrics(results)

    print("\n" + "="*60)
    print("  Results Summary")
    print("="*60)
    print(f"  Categorization accuracy  : {metrics['cat_accuracy']}%  ({metrics['cat_correct']}/{metrics['total']})")
    print(f"  Task extraction accuracy : {metrics['task_accuracy']}%  ({metrics['task_correct']}/{metrics['total']})")
    print(f"  Macro Precision          : {metrics['macro_precision']}")
    print(f"  Macro Recall             : {metrics['macro_recall']}")
    print(f"  Macro F1 Score           : {metrics['macro_f1']}")

    print(f"\n  Per-category breakdown:")
    print(f"  {'Category':<25} {'P':>5} {'R':>5} {'F1':>5} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-'*55}")
    for cat, m in metrics["per_category"].items():
        print(
            f"  {cat:<25} {m['precision']:>5} {m['recall']:>5} "
            f"{m['f1']:>5} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
        )

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump({"metrics": metrics, "detail": results}, f, indent=2)
    print(f"\n  Results saved to eval/results.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_evaluation()