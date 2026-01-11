# 📰 News Clustering using TF-IDF, PCA, and Unsupervised Learning

## 📘 Project Overview
This project demonstrates **unsupervised text clustering** applied to news articles using modern **Natural Language Processing (NLP)** and **Machine Learning** techniques. The objective is to automatically group related news stories based on their textual similarity, without prior labeling.

The workflow leverages **TF-IDF vectorization**, **dimensionality reduction (PCA)**, and two clustering algorithms — **K-Means** and **Agglomerative Clustering** — to identify inherent patterns in unstructured textual data.

---

## 🧠 Key Features
- **Text preprocessing** including tokenization, lemmatization, and stopword removal.
- **TF-IDF representation** to capture term relevance and frequency relationships.
- **Dimensionality reduction** using PCA to visualize and improve clustering efficiency.
- **Comparison of clustering algorithms**: K-Means vs Agglomerative Clustering.
- **Silhouette Score evaluation** for objective performance comparison.
- **Data visualization** with Matplotlib and Seaborn.

---

## ⚙️ Technologies Used
- **Python 3.10+**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **scikit-learn** for vectorization, PCA, and clustering
- **NLTK** for natural language preprocessing

---

## 🚀 Project Structure
```
news-clustering/
├── sample/
│   └── news_data_sample.tsv          # Sample news dataset
├── source/
│   ├── __init__.py
│   └── main.py                       # Main project script
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
python source/main.py
```

### 3. Run the tests
```bash
pytest
```

---

## 🧭 Future Improvements
- Integrate **word embeddings** (Word2Vec, BERT) for deeper semantic understanding.
- Add **topic labeling** using NLP techniques.
- Develop an **interactive dashboard** using Streamlit or Dash for real-time exploration.

---

## ✨ Author
Developed as part of a machine learning portfolio to demonstrate unsupervised NLP capabilities and model evaluation practices.

---

## 🏷️ License
This project is released under the MIT License.
