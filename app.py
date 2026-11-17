"""
FreshHarvest AI – Multi-Model Fruit Freshness & Quality Assessment System.
Flask Application Web Server & REST API.
"""

import sys
import types
# Unconditional shim for torchvision meta registrations compatibility on Windows Python 3.13
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

import os
import io
import csv
import zipfile
import json
from datetime import datetime
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, Response, send_file, flash
)
from PIL import Image

from config import (
    UPLOAD_DIR, RESULTS_DIR, SAMPLES_DIR, DEFAULT_CLASSES,
    FRUIT_TYPES, DATA_DIR, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    SUPPORTED_ARCHITECTURES
)
from database import (
    init_db, get_predictions, get_prediction_by_id, delete_prediction,
    clear_all_predictions, get_analytics_stats, get_all_model_metrics,
    set_setting, get_setting
)
from inference import predict_single, predict_multi_model, predict_batch
from sample_data_setup import setup_demo_environment

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "freshharvest-ai-secret-key-2026"
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    """Check if file extension is supported."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_demo_samples():
    """Retrieve available sample images for the 1-click test drive."""
    samples = []
    for fruit in FRUIT_TYPES:
        fresh_p = f"samples/sample_fresh_{fruit.lower()}.jpg"
        spoiled_p = f"samples/sample_spoiled_{fruit.lower()}.jpg"
        if (STATIC_DIR / fresh_p).exists() and (STATIC_DIR / spoiled_p).exists():
            samples.append({
                "fruit": fruit,
                "fresh_image": fresh_p,
                "spoiled_image": spoiled_p
            })
    return samples

from config import STATIC_DIR

# -------------------------------------------------------------
# Web Page Routes
# -------------------------------------------------------------

@app.route('/')
def index():
    """Home / Landing overview page."""
    stats = get_analytics_stats()
    samples = get_demo_samples()
    return render_template('index.html', stats=stats, samples=samples)

@app.route('/upload')
def upload_page():
    """Single image upload and inspection page."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    """Process uploaded fruit image and redirect to results."""
    if 'image' not in request.files:
        flash("No image file provided.", "danger")
        return redirect(url_for('upload_page'))

    file = request.files['image']
    if file.filename == '':
        flash("Please select an image file to upload.", "warning")
        return redirect(url_for('upload_page'))

    if not allowed_file(file.filename):
        flash("Unsupported file format. Please upload JPG, PNG, or WEBP.", "danger")
        return redirect(url_for('upload_page'))

    model_name = request.form.get('model_name', 'multi')
    generate_gradcam = 'gradcam' in request.form
    notes = request.form.get('notes', None)

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"{timestamp}_{Path(file.filename).name}"
    save_path = UPLOAD_DIR / clean_filename
    file.save(str(save_path))

    # Run inference
    if model_name == 'multi':
        result = predict_multi_model(
            image_path_or_pil=save_path,
            generate_gradcam=generate_gradcam,
            save_to_db=True,
            notes=notes
        )
        record_id = result["record_id"]
    else:
        result = predict_single(
            image_path_or_pil=save_path,
            model_name=model_name,
            generate_gradcam=generate_gradcam,
            save_to_db=True,
            notes=notes
        )
        record_id = result["record_id"]

    return redirect(url_for('result_page', record_id=record_id))

@app.route('/quick_test')
def quick_test():
    """1-Click instant test drive from static sample images."""
    img_rel_path = request.args.get('img_path', '')
    full_path = STATIC_DIR / img_rel_path

    if not full_path.exists():
        flash("Sample image not found on server.", "danger")
        return redirect(url_for('index'))

    # Run multi-model inference
    result = predict_multi_model(
        image_path_or_pil=full_path,
        generate_gradcam=True,
        save_to_db=True,
        notes="1-Click Sample Test"
    )
    return redirect(url_for('result_page', record_id=result["record_id"]))

@app.route('/result/<int:record_id>')
def result_page(record_id):
    """View prediction assessment report and Grad-CAM visualization."""
    pred = get_prediction_by_id(record_id)
    if not pred:
        flash("Record not found.", "warning")
        return redirect(url_for('history_page'))

    # Parse multi-model summary if present
    multi_models = None
    consensus = None
    if pred.get('all_models_summary'):
        try:
            summary = json.loads(pred['all_models_summary'])
            if isinstance(summary, dict) and "models" in summary:
                multi_models = {}
                for mkey, mval in summary["models"].items():
                    multi_models[mkey] = {
                        "model_display_name": SUPPORTED_ARCHITECTURES.get(mkey, {}).get("display_name", mkey),
                        "freshness": mval.get("pred", "Fresh"),
                        "fruit_type": pred["fruit_type"],
                        "confidence": mval.get("conf", 0.0),
                        "latency_ms": mval.get("lat", 0.0),
                        "heatmap_path": f"results/cam_{mkey}_{Path(pred['filename']).stem}.jpg"
                    }
                consensus = {
                    "agreement_ratio": summary.get("agreement", "Verified"),
                    "agreement_percentage": 100 if "3/3" in summary.get("agreement", "") else 66.7
                }
        except Exception as e:
            print(f"[Result] Parsing summary error: {e}")

    return render_template(
        'result.html',
        pred=pred,
        multi_models=multi_models,
        consensus=consensus
    )

@app.route('/compare', methods=['GET', 'POST'])
def compare_page():
    """Multi-model comparative benchmark page."""
    metrics = get_all_model_metrics()
    comparison = None
    return render_template('compare.html', metrics=metrics, comparison=comparison)

@app.route('/run_compare', methods=['POST'])
def run_compare():
    """Execute live side-by-side comparison across all 3 models on uploaded image."""
    if 'image' not in request.files:
        flash("No image file uploaded.", "danger")
        return redirect(url_for('compare_page'))

    file = request.files['image']
    if file.filename == '':
        flash("Please select an image file to compare.", "warning")
        return redirect(url_for('compare_page'))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"cmp_{timestamp}_{Path(file.filename).name}"
    save_path = UPLOAD_DIR / clean_filename
    file.save(str(save_path))

    comparison = predict_multi_model(
        image_path_or_pil=save_path,
        generate_gradcam=True,
        save_to_db=True,
        notes="Live Multi-Model Benchmark"
    )
    metrics = get_all_model_metrics()
    return render_template('compare.html', metrics=metrics, comparison=comparison)

@app.route('/batch', methods=['GET', 'POST'])
def batch_page():
    """Batch upload and screening page."""
    return render_template('batch.html', batch=None)

@app.route('/batch_upload', methods=['POST'])
def handle_batch_upload():
    """Process multiple image files or a ZIP archive."""
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        flash("No files selected for batch processing.", "warning")
        return redirect(url_for('batch_page'))

    model_name = request.form.get('model_name', 'efficientnet_b0')
    batch_img_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for file in uploaded_files:
        if not file.filename:
            continue
        ext = file.filename.rsplit('.', 1)[-1].lower()

        if ext == 'zip':
            # Unpack zip archive
            zip_extract_dir = UPLOAD_DIR / f"zip_{timestamp}"
            zip_extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(str(zip_extract_dir))

            for root, _, files in os.walk(str(zip_extract_dir)):
                for fname in files:
                    if fname.rsplit('.', 1)[-1].lower() in {'png', 'jpg', 'jpeg', 'webp', 'bmp'}:
                        batch_img_paths.append(str(Path(root) / fname))
        elif ext in ALLOWED_EXTENSIONS:
            clean_name = f"batch_{timestamp}_{file.filename}"
            dest = UPLOAD_DIR / clean_name
            file.save(str(dest))
            batch_img_paths.append(str(dest))

    if not batch_img_paths:
        flash("No valid image files detected in the upload.", "danger")
        return redirect(url_for('batch_page'))

    batch_results = predict_batch(batch_img_paths, model_name=model_name)
    return render_template('batch.html', batch=batch_results)

@app.route('/history')
def history_page():
    """Inspection history table with filtering and search."""
    search = request.args.get('search', '').strip() or None
    freshness = request.args.get('freshness', 'All')
    fruit = request.args.get('fruit', 'All')

    predictions = get_predictions(
        limit=200,
        filter_freshness=freshness if freshness != 'All' else None,
        filter_fruit=fruit if fruit != 'All' else None,
        search=search
    )

    return render_template(
        'history.html',
        predictions=predictions,
        current_search=search,
        current_freshness=freshness,
        current_fruit=fruit,
        fruit_types=FRUIT_TYPES
    )

@app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    """Delete a prediction record."""
    delete_prediction(record_id)
    flash("Record removed from history.", "info")
    return redirect(url_for('history_page'))

@app.route('/clear_history', methods=['POST'])
def clear_history_action():
    """Clear all records from prediction history."""
    clear_all_predictions()
    flash("All prediction history cleared.", "success")
    return redirect(url_for('history_page'))

@app.route('/export_csv')
def export_history_csv():
    """Export prediction history as a CSV file download."""
    records = get_predictions(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ID", "Timestamp", "Filename", "Fruit Type",
        "Freshness Verdict", "Confidence %", "Model Used", "Latency (ms)", "Notes"
    ])

    for r in records:
        conf_pct = (r["confidence"] * 100) if r["confidence"] <= 1.0 else r["confidence"]
        writer.writerow([
            r["id"], r["timestamp"], r["filename"], r["fruit_type"],
            r["freshness"], round(conf_pct, 2), r["model_name"], r["latency_ms"], r.get("notes", "")
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=freshharvest_history.csv"}
    )

@app.route('/admin')
def admin_page():
    """Admin analytics dashboard and model evaluation metrics."""
    stats = get_analytics_stats()
    metrics = get_all_model_metrics()
    current_dataset = get_setting("dataset_path", str(DATA_DIR))
    return render_template(
        'admin.html',
        stats=stats,
        metrics=metrics,
        default_classes=DEFAULT_CLASSES,
        current_dataset_path=current_dataset
    )

@app.route('/admin/update_dataset', methods=['POST'])
def update_dataset_path():
    """Update configured dataset path in system settings."""
    new_path = request.form.get('dataset_path', '').strip()
    if new_path:
        set_setting("dataset_path", new_path)
        flash(f"Dataset path updated to '{new_path}'", "success")
    return redirect(url_for('admin_page'))

@app.route('/admin/trigger_eval', methods=['POST'])
def trigger_evaluation():
    """Trigger evaluation script across all 3 models."""
    try:
        from evaluate import evaluate_all_models
        dataset_path = get_setting("dataset_path", str(DATA_DIR))
        evaluate_all_models(dataset_path=dataset_path)
        flash("Model evaluation completed! Confusion matrices and metrics updated.", "success")
    except Exception as e:
        flash(f"Evaluation error: {e}", "danger")
    return redirect(url_for('admin_page'))

@app.route('/admin/trigger_setup', methods=['POST'])
def trigger_sample_setup():
    """Regenerate demo fruit dataset and UI sample assets."""
    try:
        setup_demo_environment()
        flash("Sample dataset and UI demonstration assets generated successfully.", "success")
    except Exception as e:
        flash(f"Error generating sample assets: {e}", "danger")
    return redirect(url_for('admin_page'))

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for single fruit image classification."""
    if 'file' not in request.files:
        return jsonify({"error": "No file parameter found"}), 400

    file = request.files['file']
    model_name = request.form.get('model', 'efficientnet_b0')
    generate_cam = request.form.get('gradcam', 'true').lower() == 'true'

    try:
        image = Image.open(file.stream).convert('RGB')
        res = predict_single(image, model_name=model_name, generate_gradcam=generate_cam)
        return jsonify({"status": "success", "result": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """REST API endpoint for system analytics statistics."""
    return jsonify(get_analytics_stats())

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """REST API endpoint for multi-model benchmark metrics."""
    return jsonify(get_all_model_metrics())

# -------------------------------------------------------------
# Main Application Entry Point
# -------------------------------------------------------------

if __name__ == '__main__':
    # Dynamic port detection (Port 7860 for Hugging Face Spaces, $PORT for Render, 5000 for local)
    default_port = 7860 if os.environ.get('SPACE_ID') else 5000
    port = int(os.environ.get('PORT', default_port))
    print("\n========================================================")
    print(f" FreshHarvest AI Web Application Running on Port {port}")
    print(f" Access at: http://localhost:{port}")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)

