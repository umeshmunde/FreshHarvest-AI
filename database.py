"""
Database management module for FreshHarvest AI using SQLite.
Stores prediction logs, evaluation metrics, confusion matrices, and audit statistics.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from config import DB_PATH

def get_connection():
    """Get a database connection with dict-like row access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables and default records."""
    conn = get_connection()
    cursor = conn.cursor()

    # Table: predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        filename TEXT NOT NULL,
        original_image_path TEXT NOT NULL,
        heatmap_path TEXT,
        model_name TEXT NOT NULL,
        fruit_type TEXT NOT NULL,
        freshness TEXT NOT NULL,
        confidence REAL NOT NULL,
        latency_ms REAL NOT NULL,
        all_models_summary TEXT,
        is_batch INTEGER DEFAULT 0,
        batch_id TEXT,
        notes TEXT
    )
    """)

    # Table: model_metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        model_key TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        accuracy REAL NOT NULL,
        precision REAL NOT NULL,
        recall REAL NOT NULL,
        f1_score REAL NOT NULL,
        test_loss REAL,
        evaluation_date TEXT NOT NULL,
        confusion_matrix_json TEXT,
        class_report_json TEXT
    )
    """)

    # Table: system_settings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

def save_prediction(filename, original_image_path, heatmap_path, model_name,
                    fruit_type, freshness, confidence, latency_ms,
                    all_models_summary=None, is_batch=0, batch_id=None, notes=None):
    """Save an inference prediction record to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(all_models_summary, dict) or isinstance(all_models_summary, list):
        summary_str = json.dumps(all_models_summary)
    else:
        summary_str = all_models_summary or ""

    cursor.execute("""
    INSERT INTO predictions (
        timestamp, filename, original_image_path, heatmap_path,
        model_name, fruit_type, freshness, confidence, latency_ms,
        all_models_summary, is_batch, batch_id, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, filename, original_image_path, heatmap_path,
        model_name, fruit_type, freshness, float(confidence), float(latency_ms),
        summary_str, is_batch, batch_id, notes
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_predictions(limit=100, offset=0, filter_freshness=None, filter_fruit=None, search=None):
    """Fetch prediction history with optional filtering."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if filter_freshness and filter_freshness != "All":
        query += " AND freshness = ?"
        params.append(filter_freshness)

    if filter_fruit and filter_fruit != "All":
        query += " AND fruit_type = ?"
        params.append(filter_fruit)

    if search:
        query += " AND (filename LIKE ? OR notes LIKE ? OR fruit_type LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results

def get_prediction_by_id(record_id):
    """Fetch a single prediction by its primary key ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_prediction(record_id):
    """Delete a prediction record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_all_predictions():
    """Clear all records in the predictions table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

def save_model_metrics(model_key, model_name, accuracy, precision, recall, f1_score,
                       test_loss=0.0, confusion_matrix=None, class_report=None):
    """Save or update evaluation benchmark metrics for a model."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cm_json = json.dumps(confusion_matrix) if confusion_matrix is not None else "[]"
    cr_json = json.dumps(class_report) if class_report is not None else "{}"

    cursor.execute("""
    INSERT INTO model_metrics (
        model_key, model_name, accuracy, precision, recall, f1_score,
        test_loss, evaluation_date, confusion_matrix_json, class_report_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(model_key) DO UPDATE SET
        model_name=excluded.model_name,
        accuracy=excluded.accuracy,
        precision=excluded.precision,
        recall=excluded.recall,
        f1_score=excluded.f1_score,
        test_loss=excluded.test_loss,
        evaluation_date=excluded.evaluation_date,
        confusion_matrix_json=excluded.confusion_matrix_json,
        class_report_json=excluded.class_report_json
    """, (
        model_key, model_name, float(accuracy), float(precision), float(recall), float(f1_score),
        float(test_loss), now_str, cm_json, cr_json
    ))
    conn.commit()
    conn.close()

def get_all_model_metrics():
    """Retrieve evaluation metrics for all registered models."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_metrics ORDER BY accuracy DESC")
    rows = cursor.fetchall()
    conn.close()

    metrics = []
    for row in rows:
        item = dict(row)
        try:
            item["confusion_matrix"] = json.loads(item["confusion_matrix_json"])
        except Exception:
            item["confusion_matrix"] = []
        try:
            item["class_report"] = json.loads(item["class_report_json"])
        except Exception:
            item["class_report"] = {}
        metrics.append(item)
    return metrics

def get_analytics_stats():
    """Calculate aggregated stats for admin dashboard and reporting."""
    conn = get_connection()
    cursor = conn.cursor()

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    # Fresh vs Spoiled
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE freshness = 'Fresh'")
    fresh_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE freshness = 'Spoiled'")
    spoiled_count = cursor.fetchone()[0]

    # Average confidence & latency
    cursor.execute("SELECT AVG(confidence), AVG(latency_ms) FROM predictions")
    avg_conf, avg_lat = cursor.fetchone()
    avg_confidence = round(float(avg_conf or 0) * 100, 1)
    avg_latency = round(float(avg_lat or 0), 1)

    # Fruit type breakdown
    cursor.execute("""
    SELECT fruit_type, COUNT(*) as count,
           SUM(CASE WHEN freshness = 'Fresh' THEN 1 ELSE 0 END) as fresh,
           SUM(CASE WHEN freshness = 'Spoiled' THEN 1 ELSE 0 END) as spoiled
    FROM predictions
    GROUP BY fruit_type
    ORDER BY count DESC
    """)
    fruit_breakdown = [dict(row) for row in cursor.fetchall()]

    # Model usage distribution
    cursor.execute("""
    SELECT model_name, COUNT(*) as count, AVG(confidence) as avg_conf, AVG(latency_ms) as avg_lat
    FROM predictions
    GROUP BY model_name
    """)
    model_usage = [dict(row) for row in cursor.fetchall()]

    # Recent 7-day trend
    cursor.execute("""
    SELECT substr(timestamp, 1, 10) as day, COUNT(*) as total,
           SUM(CASE WHEN freshness = 'Fresh' THEN 1 ELSE 0 END) as fresh,
           SUM(CASE WHEN freshness = 'Spoiled' THEN 1 ELSE 0 END) as spoiled
    FROM predictions
    GROUP BY day
    ORDER BY day DESC
    LIMIT 7
    """)
    trend_rows = cursor.fetchall()
    history_trend = [dict(r) for r in reversed(trend_rows)]

    conn.close()

    return {
        "total_predictions": total_predictions,
        "fresh_count": fresh_count,
        "spoiled_count": spoiled_count,
        "fresh_percentage": round((fresh_count / total_predictions * 100), 1) if total_predictions > 0 else 0,
        "spoiled_percentage": round((spoiled_count / total_predictions * 100), 1) if total_predictions > 0 else 0,
        "avg_confidence": avg_confidence,
        "avg_latency": avg_latency,
        "fruit_breakdown": fruit_breakdown,
        "model_usage": model_usage,
        "history_trend": history_trend
    }

def set_setting(key, value):
    """Store or update a key-value system setting."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO system_settings (key, value, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
    """, (key, str(value), now_str))
    conn.commit()
    conn.close()

def get_setting(key, default=None):
    """Get a system setting value."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else default

# Initialize DB on module import
init_db()

