import sys, os.path
import string
import numpy as np
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC
from whoosh.analysis import StemmingAnalyzer, RegexTokenizer, LowercaseFilter, StopFilter
from whoosh.writing import BufferedWriter
from whoosh import qparser
from whoosh.qparser import QueryParser

from matrices.indexing import INDEX_DATA, INDEX_DATA_SAMPLE, INDEX_JACCARD, INDEX_WORD2VEC

def get_filenames(path_resources):
    file_count = 0
    for (path_name, _, filenames) in os.walk(path_resources):
        for filename in filenames:
            extension = filename[-4:]
            if extension == '.txt':
                file_count += 1
                path_filename = os.path.join(path_name, filename)
                yield path_filename

def get_target_text(pfilename):
    fin = open(pfilename, "r", encoding="utf-8")
    indexdoc, indexdoc_tmp, title, text = u"", u"", u"", u""
    for lf in fin:
        ldoc = lf.strip("\n")
        if ldoc:
            if ldoc[:2] == "#*": # Title 
                title = ldoc[2:]
            if ldoc[:6] == "#index": # Index 
                indexdoc_tmp = ldoc[1:]
            if ldoc[:2] == "#!": # Abstract 
                text = ldoc[2:]
        elif indexdoc_tmp and indexdoc != indexdoc_tmp:
            indexdoc = indexdoc_tmp
            yield indexdoc, title, title if not text else title + " " + text
            title, text = u"", u""

def create_bow_from_files_in(documents_path): 
    data_batch = 50000

    whoosh_index = documents_path + INDEX_DATA
    if not os.path.exists(whoosh_index):
        os.mkdir(whoosh_index)

    schema = Schema(indexdoc=ID(stored=True), 
                        title=TEXT(analyzer=StemmingAnalyzer(), stored=True), 
                        content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
                        bag_of_words=KEYWORD(stored=True, scorable=True),
                        bag_of_words_title=KEYWORD(stored=True, scorable=True),
                        cardinality=NUMERIC(int, 32, signed=False, stored=True))
    ix = create_in(whoosh_index, schema)
    writer = ix.writer(limitmb=2048)
    analizer = RegexTokenizer() | LowercaseFilter() | StopFilter()

    for pfilename in get_filenames(documents_path):
        if pfilename[-3:] != "txt":
            continue
        i = 1
        for indexdoc, title, text in get_target_text(pfilename):
            tokens = set([t.text for t in analizer(text)])
            tokens_title = " ".join(set([t.text for t in analizer(title)]))
            cardinality = len(tokens)
            tokens = " ".join(tokens)
            writer.add_document(indexdoc=indexdoc, 
                        title=title, 
                        content=text, 
                        bag_of_words=tokens,
                        bag_of_words_title=tokens_title,
                        cardinality=cardinality)
            if i % data_batch == 0:
                print("N: ", i)
            i += 1
    writer.commit()

def get_cursor(index_data_path):
    print(index_data_path)
    if os.path.exists(index_data_path):
        print(index_data_path)
        ix_data = open_dir(index_data_path)
        return ix_data

def cur_indexed_docs(data_path):
    index_data_path = data_path + INDEX_DATA
    return get_cursor(index_data_path)

def cur_indexed_sample(data_path):
    index_data_path = data_path + INDEX_DATA_SAMPLE
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
    return result[0]

def find_in_content(ix_data, query, doc_limit=10):
    parser = QueryParser("content", ix_data.schema)
    q = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q, limit = doc_limit)
    return result

def find_in_bag_of_words(ix_data, query, doc_limit=10):
    parser = QueryParser("bag_of_words", ix_data.schema)
    q = parser.parse(query)
    with ix_data.searcher() as searcher:
        result = searcher.search(q, limit = doc_limit)
    return result
