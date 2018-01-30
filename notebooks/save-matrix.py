import sys
import resources.dataset as rd
import measures.jaccard as mjaccard

if __name__ == "__main__":
    generic_name_bow = "/bag_of_words_by_document.npy"
    print("Main from command-line.", file=sys.stderr)

    documents_path = sys.argv if sys.argv[1:2] else "./resources/test"
    print("Creating similarity matrix ...")
    # J = mjaccard.create_similarity_matrix(documents_path,)

