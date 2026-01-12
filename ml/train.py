import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# FIX 1: Import directly from preprocess since they are in the same folder
from preprocess import preprocess_text 

def train_model():
    # Pathing: Ensure we look for data from the project root
    # Use relative paths from the root
    data_path = "../data/news_data.tsv"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find {data_path}. Make sure you run this from the project root.")
        return

    print("Reading data...")
    # Loading columns 1 and 2 (adjust indices if your TSV differs)
    df = pd.read_table(data_path, delimiter="\t", header=None)[[1, 2]]
    df.columns = ["label", "news_text"]
    
    # Preprocessing
    print("Preprocessing text...")
    df['cleaned'] = df['news_text'].apply(preprocess_text)
    
    # Vectorization
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned']).toarray()
    X = normalize(X)
    
    # PCA
    print("Reducing dimensions...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # KMeans
    print("Clustering...")
    model = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    model.fit(X_pca)
    
    # Save artifacts
    # FIX 2: Ensure we save to the 'models' folder in the project root
    os.makedirs("models", exist_ok=True)
    pickle.dump(vectorizer, open("../models/tfidf_vectorizer.pkl", "wb"))
    pickle.dump(pca, open("../models/pca_model.pkl", "wb"))
    pickle.dump(model, open("../models/kmeans_model.pkl", "wb"))
    
    print("✅ Models saved successfully in /models folder!")

if __name__ == "__main__":
    train_model()