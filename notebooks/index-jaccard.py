import sys
import resources.dataset as rd
import measures.jaccard as mjaccard

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
    data_path = sys.argv[1] if sys.argv[1:2] else "./resources/test"
    print("Creating similarity matrix ...")
    #mjaccard.test_whoosh(data_path)
    mjaccard.index_similarities(data_path)
    #J = mjaccard.create_similarity_matrix(documents_path)

