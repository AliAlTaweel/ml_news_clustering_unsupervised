import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Download only if necessary
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    # Lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = nltk.word_tokenize(text)
    
    # Filtering: Stopwords, length check, and stemming
    cleaned_tokens = [
        ps.stem(word) for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(cleaned_tokens)