"""
MLflow experiment tracking.
Tracks:
  - Model versions with full metrics
  - Training data snapshots
  - Hyperparameters
  - Feature importance
  - Comparison between versions
"""
import os
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "aeoa_priority_model"

_mlflow_available = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _mlflow_available = True
except ImportError:
    logger.warning("mlflow not installed — experiment tracking disabled")


def get_or_create_experiment() -> str | None:
    """Get or create MLflow experiment. Returns experiment_id."""
    if not _mlflow_available:
        return None
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp:
            return exp.experiment_id
        return mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"system": "aeoa", "task": "email_priority"},
        )
    except Exception as e:
        logger.warning(f"MLflow experiment setup failed: {type(e).__name__}")
        return None


def log_training_run(
    model_data: dict,
    metrics: dict,
    feature_names: list[str],
    n_samples: int,
) -> str | None:
    """
    Log a model training run to MLflow.
    Returns run_id or None.
    """
    if not _mlflow_available:
        return None

    try:
        exp_id = get_or_create_experiment()
        with mlflow.start_run(experiment_id=exp_id) as run:

            # Log hyperparameters
            mlflow.log_params({
                "model_type":    "LogisticRegression",
                "max_iter":      1000,
                "C":             1.0,
                "class_weight":  "balanced",
                "n_features":    len(feature_names),
                "n_samples":     n_samples,
                "feature_names": ",".join(feature_names),
            })

            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    mlflow.log_metric(key, value)

            # Log model version tag
            version = model_data.get("version","unknown")
            mlflow.set_tags({
                "version":     version,
                "system":      "aeoa",
                "model_type":  "batch_lr",
            })

            # Log model pickle as artifact
            model_path = model_data.get("model_path","")
            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")

            run_id = run.info.run_id
            logger.info(f"MLflow run logged | run_id={run_id} version={version}")
            return run_id

    except Exception as e:
        logger.warning(f"MLflow logging failed: {type(e).__name__}")
        return None


def log_online_update(
    email_id: str,
    user_id: str,
    priority: int,
    trust_score: float,
    success: bool,
):
    """Log an online learning update event."""
    if not _mlflow_available:
        return
    try:
        exp_id = get_or_create_experiment()
        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=f"online_update_{email_id[:8]}",
        ) as run:
            mlflow.log_params({
                "update_type": "online",
                "email_id":    email_id,
                "user_id":     user_id,
            })
            mlflow.log_metrics({
                "priority":    priority,
                "trust_score": trust_score,
                "success":     int(success),
            })
            mlflow.set_tag("model_type", "online_lr")
    except Exception as e:
        logger.debug(f"MLflow online update log failed: {type(e).__name__}")


def get_best_run() -> dict | None:
    """Get the best run from MLflow by test_accuracy."""
    if not _mlflow_available:
        return None
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.model_type = 'batch_lr'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=1,
        )
        if runs.empty:
            return None
        return runs.iloc[0].to_dict()
    except Exception as e:
        logger.warning(f"MLflow search failed: {type(e).__name__}")
        return None


def get_run_history(limit: int = 10) -> list[dict]:
    """Get recent training run history from MLflow."""
    if not _mlflow_available:
        return []
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.model_type = 'batch_lr'",
            order_by=["start_time DESC"],
            max_results=limit,
        )
        if runs.empty:
            return []
        return [
            {
                "run_id":        r.get("run_id",""),
                "version":       r.get("tags.version",""),
                "test_accuracy": r.get("metrics.test_accuracy"),
                "n_samples":     r.get("params.n_samples"),
                "started_at":    str(r.get("start_time","")),
            }
            for _, r in runs.iterrows()
        ]
    except Exception as e:
        logger.warning(f"MLflow history fetch failed: {type(e).__name__}")
        return []