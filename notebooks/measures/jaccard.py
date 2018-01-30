import numpy as np
from scipy.sparse import dok_matrix
import os

def jaccard_index(set_A, set_B):
    card_intersection_A_B = np.float32(len(set_A['set'] & set_B['set']))
    return card_intersection_A_B /(set_A['cardinality'] + set_B['cardinality'] - card_intersection_A_B)

# def jaccard_distance(set_A, set_B):
#    return 1.0 - jaccard_index(set_A, set_B)

def create_similarity_matrix(matrix_path, bag_of_words):
    N = len(bag_of_words)
    dmatrix = dok_matrix((N, N), dtype=np.float32)
    i = 0
    while i < N:
        j = i
        if i != j and i % 1000 == 0:
            print("d(i) :", i)
        while j < N:
            jd = jaccard_index(bag_of_words[i], bag_of_words[j])
            if jd != 0.0:
                dmatrix[i,j] = jd 
            j += 1
        i += 1
    return dmatrix
