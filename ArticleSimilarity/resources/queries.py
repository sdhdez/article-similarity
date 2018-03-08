import os
from whoosh.index import open_dir
from whoosh import qparser
from whoosh.qparser import QueryParser

from resources.dataset import INDEX_DATA

# Methods to query data 
def get_cursor(index_data_path):
    if os.path.exists(index_data_path):
        ix_data = open_dir(index_data_path)
        return ix_data

def cur_indexed_docs(data_path):
    index_data_path = data_path + INDEX_DATA
    return get_cursor(index_data_path)

def cur_indexed_sample(data_path, nrand=100, doc_limit=10):
    index_data_path = data_path + INDEX_DATA_SAMPLE + "-%d-%d" % (nrand, doc_limit)
    return get_cursor(index_data_path)

def cur_indexed_jaccard(data_path):
    index_data_path = data_path + INDEX_DATA_JACCARD
    return get_cursor(index_data_path)

def cur_indexed_word2vec(data_path):
    index_data_path = data_path + INDEX_DATA_WORD2VEC
    return get_cursor(index_data_path)

def find_indexdoc(ix_data, query, doc_limit=1):
    parser = QueryParser("indexdoc", ix_data.schema)
    q = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q, limit = doc_limit)
        for r in result:
             yield r

def find_in_content(ix_data, query, doc_limit=10):
    parser = QueryParser("content", ix_data.schema)
    q = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q, limit = doc_limit)
        print("Documents found: ", len(result))
        for r in result:
             yield r

def find_in_bag_of_words(ix_data, query, doc_limit=10):
    parser = QueryParser("bag_of_words", ix_data.schema)
    q = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q, limit = doc_limit)
        print("Documents found: ", len(result))
        for r in result:
            yield r
