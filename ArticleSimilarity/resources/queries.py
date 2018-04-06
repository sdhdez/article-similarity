"""Module to query documents"""
import os
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

from resources.dataset import INDEX_DATA

def get_cursor(index_data_path):
    """Return Whoosh cursor to given path"""
    ix_data = None
    if os.path.exists(index_data_path):
        ix_data = open_dir(index_data_path)
    return ix_data

def cur_indexed_docs(data_path):
    """"Return cursor to indexed documents"""
    index_data_path = data_path + INDEX_DATA
    return get_cursor(index_data_path)

def find_indexdoc(ix_data, query, doc_limit=1):
    """Yield documents from Whoosh index matching query"""
    parser = QueryParser("indexdoc", ix_data.schema)
    q_docs = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q_docs, limit=doc_limit)
        for result in result:
            yield result

def find_in_content(ix_data, query, doc_limit=10):
    """Yield documents with content matching query"""
    parser = QueryParser("content", ix_data.schema)
    q_docs = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q_docs, limit=doc_limit)
        print("Documents found: ", len(result))
        for result in result:
            yield result

def find_in_bag_of_words(ix_data, query, doc_limit=10):
    """Yield documents matching given words in query"""
    parser = QueryParser("bag_of_words", ix_data.schema)
    q_docs = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q_docs, limit=doc_limit)
        print("Documents found: ", len(result))
        for result in result:
            yield result
