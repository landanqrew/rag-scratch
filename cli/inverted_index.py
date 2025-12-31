from text_processing import normalize_text, tokenize, make_case_insensitive, remove_stopwords, stem
import pickle
import json
import os
from collections import Counter

class InvertedIndex:
  def __init__(self):
    self.index: dict[str, list[int]] = {}
    self.docmap: dict[int, dict[str, int | str]] = {}
    self.term_frequency: dict[int, dict[str, int]] = {}


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
  
  def get_term_frequency(self, doc_id: int, term: str) -> dict[str, int]:
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
  
  def build(self, documents: list[dict[str, int | str]]) -> None:
    """
    Build the inverted index from the documents.
    """
    for doc in documents:
      self.docmap[doc["id"]] = doc
      self.__add_document(doc["id"], f'{doc["title"]} {doc["description"]}')

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

  def load(self) -> None:
    """
    Load the inverted index from the cache directory.
    """
    path = "./cache"
    if not os.path.exists(path):
      raise NotADirectoryError(f"There is no directory at path '{path}'")
    
    # save index
    index_path = os.path.join(path, "index.pkl")
    if not os.path.exists(index_path):
      raise FileNotFoundError(f"There is no index file at path '{index_path}'")
    with open(index_path, 'rb') as f:
      byte_data = f.read()
      self.index = pickle.loads(byte_data)

    # save term frequency
    term_frequency_path = os.path.join(path, "term_frequency.pkl")
    if not os.path.exists(term_frequency_path):
      raise FileNotFoundError(f"There is no term frequency file at path '{term_frequency_path}'")
    with open(term_frequency_path, 'rb') as f:
      byte_data = f.read()
      self.term_frequency = pickle.loads(byte_data)

    # save docmap
    docmap_path = os.path.join(path, "docmap.pkl")
    if not os.path.exists(docmap_path):
      raise FileNotFoundError(f"There is no docmap file at path '{docmap_path}'")
    with open(docmap_path, 'rb') as f:
      byte_data = f.read()
      self.docmap = pickle.loads(byte_data)

