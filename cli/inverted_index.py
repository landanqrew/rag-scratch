from text_processing import normalize_text, tokenize, make_case_insensitive, remove_stopwords, stem
import pickle
import json
import os
from collections import Counter
import math

class InvertedIndex:
  def __init__(self):
    self.index: dict[str, list[int]] = {}
    self.docmap: dict[int, dict[str, int | str]] = {}
    self.term_frequency: dict[int, dict[str, int]] = {}
    self.avg_doc_length: float = 0


  def __add_document(self, doc_id: int, text: str) -> None:
    """
    Add a document to the inverted index.
    """
    #tokens = tokenize(make_case_insensitive(text))
    
    tokens = normalize_text(text)
    doc_counter = Counter(tokens)
    self.term_frequency[doc_id] = doc_counter
    for token in tokens:
      if token not in self.index:
        self.index[token] = []
      self.index[token].append(doc_id)

  def get_doc_by_id(self, doc_id: int) -> dict[str, str | int] | None:
      if doc_id is None or doc_id not in self.docmap:
          print(f"cannot find '{doc_id}' in docmap")
          return
      return self.docmap[doc_id]
      
      

  def get_documents(self, term: str) -> list[int]:
    """
    Get the documents that contain the term.
    """
    found: set[int] = set()
    # for token in tokenize(make_case_insensitive(term)):
    for token in normalize_text(term):
      if token in self.index:
        found.update(self.index[token])
    return sorted(list(found))
  
  def get_term_frequency(self, doc_id: int, term: str) -> int:
    """
    Get the term frequency for a document.
    """
    normalized_term = normalize_text(term)
    if len(normalized_term) == 0 or len(normalized_term) > 1:
      raise ValueError(f"Term '{term}' is not a single word")
    normalized_term = normalized_term[0]
    if normalized_term not in self.term_frequency[doc_id]:
      return 0
    return self.term_frequency[doc_id][normalized_term]
  
  def get_idf(self, term: str) -> float:
    """
    Get the inverse document frequency for a term.
    """
    tokens = normalize_text(term)
    if len(tokens) == 0 or len(tokens) > 1:
      raise ValueError(f"Term '{term}' is not a single word")
    token = tokens[0]
    return math.log((len(self.docmap.keys()) + 1) / (len(self.get_documents(token)) + 1))
  
  def get_bm25_idf(self, term: str) -> float:
    """
    Get the BM25 inverse document frequency for a term.
    """
    tokens = normalize_text(term)
    if len(tokens) == 0 or len(tokens) > 1:
      raise ValueError(f"Term '{term}' is not a single word")
    token = tokens[0]
    df = len(self.get_documents(token))
    N = len(self.docmap.keys())
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    return idf
  
  def get_doc_length(self, doc_id: int) -> int:
    """
    Get the document length, in words, used for BM25 length normalization.
    """
    doc = self.docmap[doc_id]
    return len(f'{doc["title"]} {doc["description"]}'.split())

  def build(self, documents: list[dict[str, int | str]]) -> None:
    """
    Build the inverted index from the documents.
    """
    total_doc_length: int = 0
    for doc in documents:
      self.docmap[int(doc["id"])] = doc
      self.__add_document(int(doc["id"]), f'{doc["title"]} {doc["description"]}')
      total_doc_length += self.get_doc_length(int(doc["id"]))

    self.avg_doc_length = total_doc_length / len(documents)
    # self.avg_doc_length (5110.2468) = 25551234 / 5000
    # print(f"self.avg_doc_length ({self.avg_doc_length}) = {total_doc_length} / {len(documents)}")

  def save(self) -> None:
    """
    Save the inverted index to the cache directory.
    """
    path = "./cache"
    if not os.path.exists(path):
      os.makedirs(path)
    # save index
    with open(os.path.join(path, "index.pkl"), 'wb') as f:
      pickle.dump(self.index, f)

    # save term frequency
    with open(os.path.join(path, "term_frequency.pkl"), 'wb') as f:
      pickle.dump(self.term_frequency, f)

    # save docmap
    with open(os.path.join(path, "docmap.pkl"), 'wb') as f:
      pickle.dump(self.docmap, f)

    # save avg_doc_length
    with open(os.path.join(path, "avg_doc_length.pkl"), 'wb') as f:
      pickle.dump(self.avg_doc_length, f)

  def load(self) -> None:
    """
    Load the inverted index from the cache directory.
    """
    path = "./cache"
    if not os.path.exists(path):
      raise NotADirectoryError(f"There is no directory at path '{path}'")
    
    # load index
    index_path = os.path.join(path, "index.pkl")
    if not os.path.exists(index_path):
      raise FileNotFoundError(f"There is no index file at path '{index_path}'")
    with open(index_path, 'rb') as f:
      byte_data = f.read()
      self.index = pickle.loads(byte_data)

    # load term frequency
    term_frequency_path = os.path.join(path, "term_frequency.pkl")
    if not os.path.exists(term_frequency_path):
      raise FileNotFoundError(f"There is no term frequency file at path '{term_frequency_path}'")
    with open(term_frequency_path, 'rb') as f:
      byte_data = f.read()
      self.term_frequency = pickle.loads(byte_data)

    # load docmap
    docmap_path = os.path.join(path, "docmap.pkl")
    if not os.path.exists(docmap_path):
      raise FileNotFoundError(f"There is no docmap file at path '{docmap_path}'")
    with open(docmap_path, 'rb') as f:
      byte_data = f.read()
      self.docmap = pickle.loads(byte_data)

    # load avg_doc_length
    adl_path = os.path.join(path, "avg_doc_length.pkl")
    if not os.path.exists(adl_path):
        raise FileNotFoundError(f"There is no avg_doc_length file at path '{adl_path}'")
    with open(adl_path, 'rb') as f:
        byte_data = f.read()
        self.avg_doc_length = pickle.loads(byte_data)

