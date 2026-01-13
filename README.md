# 📰 Production-Ready News Clustering API

## 📘 Project Overview

This project is a **production-ready Machine Learning API** built with **FastAPI** that performs unsupervised text clustering on news articles. It leverages **TF-IDF vectorization**, **PCA dimensionality reduction**, and **K-Means clustering** to group news headlines into semantic categories.

The system is designed for performance and scalability, serving real-time predictions via a RESTful interface.

---

## 🧠 Key Features

- **FastAPI Framework**: High-performance, easy-to-learn, fast-to-code, ready for production.
- **ML Pipeline**:
  - **Text Preprocessing**: Tokenization, lemmatization, and stopword removal (NLTK).
  - **TF-IDF Vectorization**: Converts text to numerical features.
  - **PCA**: Reduces dimensionality for efficient clustering.
  - **K-Means**: Clusters articles into distinct groups.
- **Real-time Inference**: `/predict` endpoint for classifying new articles on the fly.
- **Scalable Deployment**: Uses `uvicorn` as an ASGI server.

---

## ⚙️ Technologies Used

- **Python 3.9+**
- **FastAPI** & **Uvicorn**
- **scikit-learn** (Sklearn)
- **Pandas** & **NumPy**
- **NLTK** for natural language preprocessing
- **Pydantic** for data validation

---

## 🚀 Project Structure

```
news-clustering/
├── data/
│   └── news_data_sample.tsv          # Sample news dataset
├── src/
│   ├── __init__.py
│   └── app.py                       # Main project script
├── tests/
│   └── test_model.py                 # Basic functionality test
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

---

## 🧩 Workflow Summary

### 1. Data Loading

The project loads a sample dataset of news headlines and short articles from a TSV file. You can replace it with your own dataset.

### 2. Text Preprocessing

Each text is converted to lowercase, cleaned of non-alphabetic characters, tokenized, lemmatized, and stripped of stopwords. This ensures the model focuses on semantic meaning rather than noise.

### 3. Feature Extraction with TF-IDF

The cleaned text corpus is vectorized using TF-IDF, representing each document as a weighted vector of word importance.

### 4. Dimensionality Reduction

Principal Component Analysis (PCA) is applied to project high-dimensional TF-IDF features into 2D space for visualization and computational efficiency.

### 5. Clustering

Two algorithms are implemented and compared:

- **K-Means Clustering**: Partitions data into K clusters by minimizing intra-cluster variance.
- **Agglomerative Clustering**: A hierarchical approach that merges points iteratively based on similarity.

### 6. Evaluation

The **Silhouette Score** quantifies how well each point fits within its assigned cluster, providing an objective metric for model comparison.

### 7. Visualization

Results are visualized using scatter plots that highlight how each clustering algorithm groups the data.

---

## 📊 Example Results

After running the script, the console prints silhouette scores for both clustering methods and displays a visual plot comparing them.

Example output:

```
Silhouette Score (K-Means): 0.523
Silhouette Score (Agglomerative): 0.471
```

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the main script

```bash
python src/app.py
```

### 3. Run the tests

```bash
pytest
```
### 4. Run the server

```bash
uvicorn src.app:app --reload    
```

---

## 🧭 Future Improvements

- Integrate **word embeddings** (Word2Vec, BERT) for deeper semantic understanding.
- Add **topic labeling** using NLP techniques.
- Develop an **interactive dashboard** using Streamlit or Dash for real-time exploration.

---

## 🏷️ License

This project is released under the MIT License.
