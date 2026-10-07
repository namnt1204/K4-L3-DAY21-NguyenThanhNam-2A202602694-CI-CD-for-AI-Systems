from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.

    Ho tro ca AWS S3 (boto3) va GCP (google.cloud.storage).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bucket = os.environ.get("ARTIFACT_BUCKET", "")

    # Neu da ton tai san o local (khi test), khong bat buoc phai tai tu cloud neu khong co credential
    if not bucket and os.path.exists(MODEL_PATH):
        print(f"Su dung model san co tai: {MODEL_PATH}")
        return

    # Thu tai tu AWS S3 bang boto3
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.download_file(bucket, MODEL_KEY, MODEL_PATH)
        print("Model da duoc tai xuong tu AWS S3.")
        return
    except Exception as s3_err:
        pass

    # Thu tai tu Google Cloud Storage neu cau hinh GCP
    try:
        from google.cloud import storage
        client = storage.Client()
        b = client.bucket(bucket)
        blob = b.blob(MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print("Model da duoc tai xuong tu Google Cloud Storage.")
        return
    except Exception as gcp_err:
        pass

    # Fallback cho moi truong test cuc bo
    if os.path.exists("models/model.joblib"):
        import shutil
        shutil.copy("models/model.joblib", MODEL_PATH)
        print(f"Sao chep models/model.joblib cuc bo sang {MODEL_PATH}")
        return

    if os.path.exists(MODEL_PATH):
        print(f"Su dung model san co tai {MODEL_PATH}")
        return

    raise RuntimeError(f"Khong the tai model tu cloud storage (Bucket: {bucket}, Key: {MODEL_KEY})")


# Goi ham nay khi module duoc import (chay khi server khoi dong)
download_model()
model = joblib.load(MODEL_PATH)


FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    # TODO 6: Kiem tra so luong dac trung.
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail="Expected 10 features (adult income)"
        )

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    import pandas as pd
    df_features = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    pred = int(model.predict(df_features)[0])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
