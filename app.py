import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import normalize
# Import the helper from your ml folder
from ml.preprocess import preprocess_text

app = FastAPI(title="News Clustering API")

MODELS = {}

@app.on_event("startup")
async def load_artifacts():
    try:
        MODELS["vectorizer"] = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
        MODELS["pca"] = pickle.load(open("models/pca_model.pkl", "rb"))
        MODELS["kmeans"] = pickle.load(open("models/kmeans_model.pkl", "rb"))
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Active", "model": "KMeans Clustering"}

@app.post("/predict")
async def predict(request: NewsRequest):
    if not MODELS:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    # 1. Preprocess using your specific logic
    clean_text = preprocess_text(request.text)
    
    # 2. Transform
    X_tfidf = MODELS["vectorizer"].transform([clean_text]).toarray()
    X_norm = normalize(X_tfidf)
    X_pca = MODELS["pca"].transform(X_norm)
    
    # 3. Predict
    prediction = MODELS["kmeans"].predict(X_pca)
    
    return {
        "cluster": int(prediction[0]),
        "original_text": request.text[:100] + "..."
    }

# This allows you to also run it via 'python app.py'
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)