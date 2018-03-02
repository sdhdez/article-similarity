import os.path
import pickle
import numpy as np
from scipy.sparse import dok_matrix
from whoosh.index import open_dir, create_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC, STORED
from whoosh import qparser
from whoosh.qparser import QueryParser
from whoosh.query import Term, And

from matrices.indexing import INDEX_DATA, INDEX_DATA_SAMPLE, INDEX_JACCARD, INDEX_WORD2VEC

def get_matrix(measures, field_list):
    N = len(field_list)
    dmatrix = dok_matrix((N, N), dtype=np.float32)
    n_docs = 0
    for i, f in enumerate(field_list):
        j = i
        while j < N:
            setA = field_list[i]
            setB = field_list[j]
            if setA in measures and setB in measures[setA]:
                dmatrix[i,j] = measures[setA][setB]
                dmatrix[j,i] = measures[setA][setB]
            else:
                dmatrix[i,j] = 0
                dmatrix[j,i] = 0
            n_docs += 1
            j += 1
    print(i+1,j)
    return dmatrix.asformat('csr')

def load_matrix_jaccard_sim(data_path):
    index_jaccard_path = data_path + INDEX_JACCARD
    index_jaccard_np_matrix_name = data_path + INDEX_JACCARD + "-np-matrix"
    index_jaccard_np_matrix_file = data_path + INDEX_JACCARD + "-np-matrix.npy"
    if os.path.exists(index_jaccard_path):
        if not os.path.exists(index_jaccard_np_matrix_file):
            print("Loading indexed jaccard ...")
            ix_jaccard = open_dir(index_jaccard_path)
            with ix_jaccard.reader() as reader:
                measures = get_measures(reader)
                field_list = get_document_ids(data_path)
                print("Loading matrix ...")
                np_matrix = get_matrix(measures, field_list)
                np.save(index_jaccard_np_matrix_name, np_matrix.todense())
                return np.load(index_jaccard_np_matrix_file)
        else: 
            print("Loading saved matrix ...")
            return np.load(index_jaccard_np_matrix_file) 

def load_matrix_word2vec_sim(data_path):
    index_word2vec_path = data_path + INDEX_WORD2VEC
    index_word2vec_np_matrix_name = data_path + INDEX_WORD2VEC + '-np-matrix'
    index_word2vec_np_matrix_file = data_path + INDEX_WORD2VEC + '-np-matrix.npy'
    if os.path.exists(index_word2vec_path):
        if not os.path.exists(index_word2vec_np_matrix_file):
            print("Loading indexed word2vec ...")
            ix_word2vec = open_dir(index_word2vec_path)
            with ix_word2vec.reader() as reader:
                measures = get_measures(reader)
                field_list = get_document_ids(data_path)
                print("Loading matrix ...")
                np_matrix = get_matrix(measures, field_list)
                np.save(index_word2vec_np_matrix_name, np_matrix.todense())
                return np.load(index_word2vec_np_matrix_file) 
        else: 
            print("Loading saved matrix ...")
            return np.load(index_word2vec_np_matrix_file) 
