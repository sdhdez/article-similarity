import sys
import resources.dataset as rd
import matrices.indexing as mindexing

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
    data_path = sys.argv[1] if sys.argv[1:2] else "./resources/test"
    print("Loading jaccard ...")
    csr_matrix_jaccard = mindexing.load_matrix_jaccard_sim(data_path)
    print(csr_matrix_jaccard)
    print("Loading word2vec ...")
    csr_matrix_word2vec = mindexing.load_matrix_word2vec_sim(data_path)

