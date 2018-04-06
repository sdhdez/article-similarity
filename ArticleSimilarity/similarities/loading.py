"""Module to load similarity"""
import numpy as np

from similarities import matrices as sm
from resources import dataset as rd

def load_matrix(data_path, method="jaccard", related_docs=True):
    # Get suffix for current sample parameters
    extra_suffix = "-related" if related_docs else ""
    suffix_path = rd.get_default_suffix(extra_suffix=extra_suffix)
    # Form measures path
    measures_path = data_path + sm.MEASURES_PATH + suffix_path + ".bin"
    # Get measures
    measures = sm.load_measures(measures_path, method)
    # Get docs ids
    if related_docs:
        docs_ids = rd.get_docidssample_aminer_related(data_path)
    else:
        docs_ids = rd.get_docidssample_aminer_random(data_path)
    # Number of docs
    num_docs = len(docs_ids)
    initial_matrix = [[0 for x in range(num_docs)] for y in range(num_docs)]
    for d_i in range(0, num_docs):
        # j = i to avoid repetition, it is a symmetric matrix
        doc_i = docs_ids[d_i]
        d_j = d_i
        # Columns
        while d_j < num_docs:
            doc_j = docs_ids[d_j]
            indexdoc = doc_i + doc_j
            # print(d_i, d_j, measures[indexdoc][method])
            measure = np.float64(measures[indexdoc][method])
            initial_matrix[d_i][d_j] = measure
            if d_i != d_j:
                initial_matrix[d_j][d_i] = measure
            d_j += 1
    return initial_matrix

def load_matrix_word2vec_sim(data_path, related_docs=True):
    """Load word2vec matrix"""
    return np.array(load_matrix(data_path, method="word2vec", related_docs=related_docs))

def load_matrix_jaccard_sim(data_path, related_docs=True):
    """Load jaccard matrix"""
    return np.array(load_matrix(data_path, method="jaccard", related_docs=related_docs))
