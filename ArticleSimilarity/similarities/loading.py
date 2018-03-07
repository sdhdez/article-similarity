import os.path
import numpy as np

from similarities import matrices as sm
from resources import dataset as rd

def load_matrix(data_path, sim_type="jaccard", related_docs = True):
    # Get suffix for current sample parameters
    extra_suffix = "-related" if related_docs else ""  
    suffix_path = rd.get_default_suffix(extra_suffix=extra_suffix)
    # Form measures path 
    measures_path = data_path + sm.MEASURES_PATH + suffix_path + ".bin"
    # Get measures 
    measures = sm.load_measures(measures_path)
    # Get docs ids 
    if related_docs:
        docs_ids = rd.get_docids_sample_aminer_related(data_path)
    else:
        docs_ids = rd.get_docids_sample_aminer_random(data_path)
    # Number of docs
    N = len(docs_ids)
    A = [[0 for x in range(N)] for y in range(N)]
    for d_i in range(0, N):
        # j = i to avoid repetition, it is a symmetric matrix 
        doc_i = docs_ids[d_i]
        d_j = d_i
        # Columns
        m_cols = []
        while d_j < N:
            doc_j = docs_ids[d_j]
            indexdoc = doc_i + doc_j
            # print(d_i, d_j, measures[indexdoc][sim_type])
            measure = np.float64(measures[indexdoc][sim_type])
            A[d_i][d_j] = measure
            if d_i != d_j:
                A[d_j][d_i] = measure
            d_j+=1
    return A

def load_matrix_word2vec_sim(data_path, related_docs = True):
    return np.array(load_matrix(data_path, sim_type="word2vec", related_docs=related_docs))

def load_matrix_jaccard_sim(data_path, related_docs = True):
    return np.array(load_matrix(data_path, sim_type="jaccard", related_docs=related_docs))
