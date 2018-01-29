import numpy as np

def jaccard_index(set_A, set_B):
    card_intersection_A_B = len(set_A['set'] & set_B['set']) * 1.0
    return card_intersection_A_B /(set_A['cardinality'] + set_B['cardinality'] - card_intersection_A_B)

def distance_matrix(doc_list):
    N = len(doc_list)
    dmatrix = np.zeros([N, N])
    for A in doc_list:
        for B in doc_list:
            print(jaccard_index(doc_list[A], doc_list[B]))

    

