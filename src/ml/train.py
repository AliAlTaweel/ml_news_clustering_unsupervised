import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# Relative import
from src.ml.preprocess import preprocess_text 

def train_model():
    # Use absolute paths based on the project root (CWD)
    root_dir = os.getcwd()
    data_path = os.path.join(root_dir, "data", "news_data.tsv")
    model_dir = os.path.join(root_dir, "models")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find {data_path}")
        return

    print("--- Starting Training Pipeline ---")
    
    # 1. Load Data
    df = pd.read_table(data_path, delimiter="\t", header=None)[[1, 2]]
    df.columns = ["label", "news_text"]
    
    # 2. Preprocess
    print("Preprocessing...")
    df['cleaned'] = df['news_text'].apply(preprocess_text)
    
    # 3. Vectorize
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned']).toarray()
    X = normalize(X)
    
    # 4. Dimensionality Reduction (PCA)
    print("Reducing dimensions...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # 5. KMeans Training
    print("Clustering (KMeans)...")
    kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X_pca)
    km_score = silhouette_score(X_pca, kmeans.labels_)
    
    # 6. Agglomerative (For comparison)
    print("Clustering (Agglomerative)...")
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X_pca)
    agg_score = silhouette_score(X_pca, agg_labels)

    # 7. Save Artifacts
    os.makedirs(model_dir, exist_ok=True)
    
    artifacts = {
        "tfidf_vectorizer.pkl": vectorizer,
        "pca_model.pkl": pca,
        "kmeans_model.pkl": kmeans,
        "metrics.pkl": {"kmeans": km_score, "agglomerative": agg_score}
    }
    
    for filename, obj in artifacts.items():
        with open(os.path.join(model_dir, filename), "wb") as f:
            pickle.dump(obj, f)
    
    print(f"✅ Training Complete. KMeans Silhouette: {km_score:.4f}")
    print(f"✅ Models saved in: {model_dir}")

if __name__ == "__main__":
    train_model()