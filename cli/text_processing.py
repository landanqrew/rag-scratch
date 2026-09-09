import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def make_case_insensitive(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def tokenize(text: str) -> list[str]:
    return list(filter(lambda x: len(x) > 0, text.split()))

def remove_stopwords(tokens: list[str]) -> list[str]:
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def stem(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def normalize_text(text: str) -> list[str]:
    return remove_stopwords(stem(tokenize(remove_punctuation(make_case_insensitive(text)))))
