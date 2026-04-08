CREATE TABLE IF NOT EXISTS emails (
    id        TEXT PRIMARY KEY,
    subject   TEXT,
    body      TEXT,
    sender    TEXT,
    timestamp TEXT,
    source    TEXT DEFAULT 'demo'
);

CREATE TABLE IF NOT EXISTS processed (
    email_id      TEXT PRIMARY KEY,
    category      TEXT,
    priority      INTEGER,
    task          TEXT,
    task_type     TEXT,
    steps         TEXT,
    deadline      TEXT,
    summary       TEXT,
    confidence    REAL,
    needs_review  INTEGER DEFAULT 0,
    review_reason TEXT,
    snoozed_until TEXT,
    FOREIGN KEY (email_id) REFERENCES emails(id)
);

CREATE TABLE IF NOT EXISTS drafts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id   TEXT,
    subject    TEXT,
    body       TEXT,
    persona    TEXT,
    created_at TEXT,
    FOREIGN KEY (email_id) REFERENCES emails(id)
);

CREATE TABLE IF NOT EXISTS prompts (
    name     TEXT PRIMARY KEY,
    template TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id     TEXT,
    field        TEXT,
    old_value    TEXT,
    new_value    TEXT,
    corrected_at TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tool       TEXT,
    email_id   TEXT,
    latency_ms REAL,
    success    INTEGER,
    error_msg  TEXT,
    called_at  TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    job_type     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    total        INTEGER DEFAULT 0,
    completed    INTEGER DEFAULT 0,
    errors       INTEGER DEFAULT 0,
    progress     INTEGER DEFAULT 0,
    result_json  TEXT,
    error_msg    TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS ml_training_data (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id      TEXT NOT NULL,
    features_json TEXT NOT NULL,
    label         INTEGER NOT NULL,
    source        TEXT DEFAULT 'feedback',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     TEXT NOT NULL,
    accuracy    REAL,
    n_samples   INTEGER,
    model_path  TEXT,
    is_active   INTEGER DEFAULT 0,
    trained_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limits (
    user_id      TEXT NOT NULL,
    action       TEXT NOT NULL,
    count        INTEGER DEFAULT 0,
    window_start TEXT NOT NULL,
    PRIMARY KEY (user_id, action)
);