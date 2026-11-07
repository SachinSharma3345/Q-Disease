"""API route handlers using Starlette."""
import json
import traceback
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, Mount

from ..core import storage
from ..services import dataset_service, training_service, report_service


def _err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def _ok(data: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


# ---- Health ----

async def health(request: Request):
    st = storage.get_status()
    return _ok({
        "status": "ok",
        "version": "1.0.0",
        "storage": st,
        "disclaimer": (
            "Q-Disease is an experimental research and educational platform. "
            "Not a clinically validated medical device."
        ),
    })


# ---- Datasets ----

async def list_datasets(request: Request):
    datasets = dataset_service.list_datasets()
    return _ok({"datasets": datasets, "total": len(datasets)})


async def upload_dataset(request: Request):
    try:
        form = await request.form()
        file = form.get("file")
        if not file:
            return _err("No file uploaded.")
        content = await file.read()
        doc = dataset_service.save_dataset(file.filename, content)
        return _ok({"message": "Dataset uploaded.", "dataset": doc}, 201)
    except ValueError as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"Upload failed: {e}", 500)


async def get_dataset(request: Request):
    dataset_id = request.path_params["dataset_id"]
    doc = dataset_service.get_dataset(dataset_id)
    if not doc:
        return _err("Dataset not found.", 404)
    return _ok({"dataset": doc})


async def delete_dataset(request: Request):
    dataset_id = request.path_params["dataset_id"]
    ok = dataset_service.delete_dataset(dataset_id)
    if not ok:
        return _err("Dataset not found.", 404)
    return _ok({"message": "Deleted."})


async def configure_dataset(request: Request):
    dataset_id = request.path_params["dataset_id"]
    body = await request.json()
    target = body.get("target_column")
    exclude = body.get("exclude_columns", [])
    if not target:
        return _err("target_column is required.")
    ok = dataset_service.update_dataset_config(dataset_id, target, exclude)
    if not ok:
        return _err("Dataset not found.", 404)
    return _ok({"message": "Configuration saved."})


async def dataset_summary(request: Request):
    dataset_id = request.path_params["dataset_id"]
    try:
        summary = dataset_service.get_dataset_summary(dataset_id)
        return _ok(summary)
    except ValueError as e:
        return _err(str(e), 404)
    except Exception as e:
        return _err(f"Summary failed: {e}", 500)


# ---- Preprocessing preview ----

async def preprocessing_preview(request: Request):
    """Preview preprocessing on a dataset without training."""
    try:
        body = await request.json()
        dataset_id = body.get("dataset_id")
        target_column = body.get("target_column")
        exclude_columns = body.get("exclude_columns", [])
        preprocessing = body.get("preprocessing", {"imputation": "mean", "scaler": "standard"})
        test_size = body.get("test_size", 0.2)

        df = dataset_service.load_dataframe(dataset_id)
        from ..preprocessing.pipeline import prepare_dataset
        _, _, meta = prepare_dataset(df, target_column, exclude_columns, test_size, preprocessing)
        return _ok({"preview": meta})
    except Exception as e:
        return _err(str(e), 400)


# ---- Training ----

async def train_model(request: Request):
    try:
        job = await request.json()
        result = training_service.run_training(job)
        return _ok(result, 201)
    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        traceback.print_exc()
        return _err(f"Training failed: {e}", 500)


# ---- Experiments ----

async def list_experiments(request: Request):
    exps = storage.find_all("experiments")
    return _ok({"experiments": exps, "total": len(exps)})


async def get_experiment(request: Request):
    eid = request.path_params["experiment_id"]
    exp = storage.find_one("experiments", {"_id": eid})
    if not exp:
        return _err("Experiment not found.", 404)
    return _ok({"experiment": exp})


async def experiment_comparison(request: Request):
    eid = request.path_params["experiment_id"]
    exp = storage.find_one("experiments", {"_id": eid})
    if not exp:
        return _err("Experiment not found.", 404)
    model_doc = training_service.get_model(exp.get("model_id", ""))
    if not model_doc:
        return _err("Model not found.", 404)
    return _ok({
        "experiment": exp,
        "metrics": model_doc.get("metrics"),
        "roc_curve": model_doc.get("roc_curve"),
        "explainability": model_doc.get("explainability"),
        "circuit_info": model_doc.get("circuit_info"),
        "preprocessing_meta": model_doc.get("preprocessing_meta"),
    })


# ---- Models ----

async def list_models(request: Request):
    models = training_service.list_models()
    return _ok({"models": models, "total": len(models)})


async def get_model(request: Request):
    mid = request.path_params["model_id"]
    doc = training_service.get_model(mid)
    if not doc:
        return _err("Model not found.", 404)
    return _ok({"model": doc})


async def model_metrics(request: Request):
    mid = request.path_params["model_id"]
    doc = training_service.get_model(mid)
    if not doc:
        return _err("Model not found.", 404)
    return _ok({"metrics": doc.get("metrics"), "roc_curve": doc.get("roc_curve")})


async def model_explanations(request: Request):
    mid = request.path_params["model_id"]
    doc = training_service.get_model(mid)
    if not doc:
        return _err("Model not found.", 404)
    return _ok({"explainability": doc.get("explainability"), "feature_names": doc.get("feature_names")})


# ---- Predictions ----

async def predict_single(request: Request):
    try:
        body = await request.json()
        model_id = body.get("model_id")
        features = body.get("features", {})
        if not model_id:
            return _err("model_id required.")
        result = training_service.predict_single(model_id, features)
        return _ok(result)
    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        return _err(f"Prediction failed: {e}", 500)


async def predict_batch(request: Request):
    try:
        form = await request.form()
        model_id = form.get("model_id")
        file = form.get("file")
        if not model_id or not file:
            return _err("model_id and file required.")
        content = await file.read()
        result = training_service.predict_batch(model_id, content)
        return _ok(result)
    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        return _err(f"Batch prediction failed: {e}", 500)


async def list_predictions(request: Request):
    preds = storage.find_all("predictions")
    return _ok({"predictions": preds, "total": len(preds)})


# ---- Reports ----

async def report_json(request: Request):
    eid = request.path_params["experiment_id"]
    try:
        data = report_service.generate_json_report(eid)
        return Response(
            content=data,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=report_{eid[:8]}.json"},
        )
    except ValueError as e:
        return _err(str(e), 404)


async def report_csv(request: Request):
    eid = request.path_params["experiment_id"]
    try:
        data = report_service.generate_csv_report(eid)
        return Response(
            content=data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=report_{eid[:8]}.csv"},
        )
    except ValueError as e:
        return _err(str(e), 404)


# ---- Dashboard stats ----

async def dashboard_stats(request: Request):
    return _ok({
        "total_datasets": storage.count("datasets"),
        "total_experiments": storage.count("experiments"),
        "total_models": storage.count("models"),
        "total_predictions": storage.count("predictions"),
        "recent_experiments": storage.find_all("experiments", limit=5),
        "recent_models": storage.find_all("models", limit=5),
    })


# ---- Route table ----

routes = [
    Route("/api/health", health),
    Route("/api/dashboard", dashboard_stats),

    Route("/api/datasets", list_datasets),
    Route("/api/datasets/upload", upload_dataset, methods=["POST"]),
    Route("/api/datasets/{dataset_id}", get_dataset),
    Route("/api/datasets/{dataset_id}", delete_dataset, methods=["DELETE"]),
    Route("/api/datasets/{dataset_id}/configure", configure_dataset, methods=["POST"]),
    Route("/api/datasets/{dataset_id}/summary", dataset_summary),

    Route("/api/preprocessing/preview", preprocessing_preview, methods=["POST"]),

    Route("/api/models/train", train_model, methods=["POST"]),
    Route("/api/models", list_models),
    Route("/api/models/{model_id}", get_model),
    Route("/api/models/{model_id}/metrics", model_metrics),
    Route("/api/models/{model_id}/explanations", model_explanations),

    Route("/api/experiments", list_experiments),
    Route("/api/experiments/{experiment_id}", get_experiment),
    Route("/api/experiments/{experiment_id}/comparison", experiment_comparison),

    Route("/api/predict", predict_single, methods=["POST"]),
    Route("/api/predict/batch", predict_batch, methods=["POST"]),
    Route("/api/predictions", list_predictions),

    Route("/api/reports/{experiment_id}/json", report_json),
    Route("/api/reports/{experiment_id}/csv", report_csv),
]
