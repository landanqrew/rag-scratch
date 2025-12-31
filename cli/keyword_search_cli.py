#!/usr/bin/env python3

import argparse
import json
import math
from text_processing import normalize_text
from inverted_index import InvertedIndex
from solution.keyword_search import build_command, search_command, tf_command, InvertedIndex as SolutionInvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            if not args.query:
                print("Please provide a search query")
                return
            # print the search query here
            print(f"Searching for: {args.query}")
            index = load()
            '''movies = get_movies()
            results = title_keyword_search(args.query, movies)'''
            doc_indexes = index.get_documents(args.query)
            for i, result in enumerate(doc_indexes):
                print(f"{i+1}. {index.docmap[result]['title']}")
        case "build":
            index = build()

        case "tf":
            if not args.doc_id or not args.term:
                print("Please provide a document ID and term")
                return
            # print(f"Term frequency for document {args.doc_id} and term {args.term} is {get_term_frequency(args.doc_id, args.term)}")
            print(get_term_frequency(args.doc_id, args.term))    

        case "idf":
            if not args.term:
                print("Please provide a term")
                return
            idf = get_inverse_document_frequency(args.term)
            print(print(f"Inverse document frequency of '{args.term}': {idf:.2f}"))
            
        case _:
            parser.print_help()

def get_movies() -> list[dict]:
    with open("data/movies.json", "r") as f:
        return json.load(f)['movies']

def exact_title_keyword_search(query: str, movies: list[dict]) -> list[dict]:
    return [m for m in movies if query in m["title"]]

def title_keyword_search(query: str, movies: list[dict]) -> list[dict]:
    normalized_query = normalize_text(query)
    # print(f"Normalized query: {normalized_query}")
    results: list[dict] = []
    for movie in movies:
        normalized_title = normalize_text(movie["title"])
        # print(f"Normalized title: {normalized_title}")
        matched: bool = False
        for token in normalized_query:
            for title_token in normalized_title:
                if token in title_token:
                    matched = True
                    break

        if matched:
            results.append(movie)
        
    return results

def get_term_frequency(doc_id: int, term: str) -> int:
    index = load()
    return index.get_term_frequency(doc_id, term)

def get_inverse_document_frequency(term: str) -> float:
    tokens = normalize_text(term)
    if len(tokens) == 0 or len(tokens) > 1:
      raise ValueError(f"Term '{term}' is not a single word")
    token = tokens[0]
    index = load()
    total_doc_count = len(index.docmap.keys())
    term_match_doc_count = len(index.get_documents(token))
    # print("token:", token)
    print("total_doc_count:", total_doc_count)
    print("term_match_doc_count:", term_match_doc_count)
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def build() -> InvertedIndex:
    index = SolutionInvertedIndex()
    index.build()
    index.save()
    # docs = index.get_documents()
    # print(f"First document for token 'merida' = {docs[0]}")
    print(f"Index built and saved to cache directory")
    return index

def load() -> InvertedIndex:
    index = InvertedIndex()
    index.load()
    return index

    

if __name__ == "__main__":
    main()