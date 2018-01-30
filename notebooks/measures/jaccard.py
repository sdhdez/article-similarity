import numpy as np
from scipy.sparse import dok_matrix

def jaccard_index(set_A, set_B):
    card_intersection_A_B = np.float32(len(set_A['set'] & set_B['set']))
    return card_intersection_A_B /(set_A['cardinality'] + set_B['cardinality'] - card_intersection_A_B)

# def jaccard_distance(set_A, set_B):
#    return 1.0 - jaccard_index(set_A, set_B)

def similarity_matrix(doc_list):
    N = len(doc_list)
    dmatrix = dok_matrix((N, N), dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N:
            jd = jaccard_index(doc_list[i], doc_list[j])
            if jd != 0.0:
                dmatrix[i,j] = jd 
                if i != j:
                    dmatrix[j,i] = jd 
            j += 1
        i += 1
    return dmatrix
