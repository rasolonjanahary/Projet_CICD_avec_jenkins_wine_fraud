from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import Request
from pydantic import BaseModel
import pandas as pd
import joblib
import time
import os

app = FastAPI(title="Wine Prediction API")

Instrumentator().instrument(app).expose(app)

REQUEST_PREDICTION = Counter(
    "prediction_requests_total",
    "Nombre total de prédictions"
)

PREDICTION_RESULT = Counter(
    "prediction_result_total",
    "Nombre de prédictions par classe",
    ["prediction"]
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Nombre d'erreurs durant la prédiction"
)

PREDICTION_TIME = Histogram(
    "prediction_duration_seconds",
    "Temps d'inférence du modèle"
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Le modèle est chargé"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WiForm(BaseModel):
    fixed_acidity       :float
    volatile_acidity    :float
    citric_acid         :float
    residual_sugar      :float
    chlorides           :float
    free_sulfur_dioxide :float
    total_sulfur_dioxide:float
    density             :float
    pH                  :float
    sulphates           :float
    alcohol             :float 
    type                :int


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"..", "models", "model_dc.pkl")
SCALER_PATH = os.path.join(BASE_DIR,"..", "models", "scaler1.pkl")
TEMPLATES_PATH = os.path.join(BASE_DIR, "templates")

model = joblib.load(MODEL_PATH)
sc = joblib.load(SCALER_PATH)

MODEL_LOADED.set(1)

templates = Jinja2Templates(directory=TEMPLATES_PATH)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api")
def home():
    return {"message": "API OK"}

@app.get("/health")
def health():
    return {
        "status": "UP"
    }

@app.get("/ready")
def ready():
    return {
        "model_loaded": True
    }

@app.post("/predict")
def predict(data: WiForm):

    REQUEST_PREDICTION.inc()

    start = time.perf_counter()

    try:

        input_dict = {
            "fixed_acidity": data.fixed_acidity,
            "volatile_acidity": data.volatile_acidity,
            "citric_acid": data.citric_acid,
            "residual_sugar": data.residual_sugar,
            "chlorides": data.chlorides,
            "free_sulfur_dioxide": data.free_sulfur_dioxide,
            "total_sulfur_dioxide": data.total_sulfur_dioxide,
            "density": data.density,
            "pH": data.pH,
            "sulphates": data.sulphates,
            "alcohol": data.alcohol,
            "type": data.type,
        }

        df = pd.DataFrame([input_dict])

        xc = sc.transform(df)

        pred = model.predict(xc)

        prediction = int(pred[0])

        PREDICTION_RESULT.labels(
            prediction=str(prediction)
        ).inc()

        return {
            "prediction": prediction
        }

    except Exception:

        PREDICTION_ERRORS.inc()

        raise

    finally:

        elapsed = time.perf_counter() - start

        PREDICTION_TIME.observe(elapsed)