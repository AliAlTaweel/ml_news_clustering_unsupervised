import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
    return " ".join(tokens)