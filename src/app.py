import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import normalize

# Import from the local ml module
from src.ml.preprocess import preprocess_text

app = FastAPI(title="News Clustering API")

# Global dictionary to store models
MODELS = {}

@app.on_event("startup")
async def load_artifacts():
    """Load models from the root/models directory on startup."""
    # We look for models relative to the project root
    base_path = os.path.join(os.getcwd(), "models")
    
    try:
        MODELS["vectorizer"] = pickle.load(open(f"{base_path}/tfidf_vectorizer.pkl", "rb"))
        MODELS["pca"] = pickle.load(open(f"{base_path}/pca_model.pkl", "rb"))
        MODELS["kmeans"] = pickle.load(open(f"{base_path}/kmeans_model.pkl", "rb"))
        MODELS["metrics"] = pickle.load(open(f"{base_path}/metrics.pkl", "rb"))
        print("✅ All ML artifacts loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Critical Error: Model files not found in {base_path}. Did you run train.py?")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {
        "status": "Active", 
        "model": "KMeans Clustering",
        "metrics": MODELS.get("metrics", "Not available")
    }

@app.post("/predict")
async def predict(request: NewsRequest):
    if not MODELS:
        raise HTTPException(status_code=500, detail="Models not initialized. Check server logs.")
    
    # 1. Preprocess
    clean_text = preprocess_text(request.text)
    
    # 2. Vectorize and Normalize
    # Transform expects a list/array
    X_tfidf = MODELS["vectorizer"].transform([clean_text]).toarray()
    X_norm = normalize(X_tfidf)
    
    # 3. Dimensionality Reduction
    X_pca = MODELS["pca"].transform(X_norm)
    
    # 4. Predict
    prediction = MODELS["kmeans"].predict(X_pca)
    
    return {
        "cluster": int(prediction[0]),
        "input_snippet": request.text[:100] + "...",
        "model_info": {
            "silhouette_score": MODELS.get("metrics", {}).get("kmeans"),
            "n_clusters": MODELS["kmeans"].n_clusters
        }
    }