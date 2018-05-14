"""Load run needed methods to load resources a measure similarities"""
import sys
from somhos.resources import dataset as sd
from somhos.similarities import matrices as sm
from somhos.similarities import loading as sl

def main():
    """Testing package"""
    print("Main from command-line.", file=sys.stderr)
    data_path = sys.argv[1] if sys.argv[1:2] else "./src/somhos/resources/test"
    print("Generating indexes and matrices ...", file=sys.stderr)
    print("\nRelated documents", file=sys.stderr)
    sd.index_aminer_txt_in(data_path)
    sd.save_sample_aminer_related(data_path)
    sm.measures_sample_aminer_related(data_path)
    print("\nRandom documents", file=sys.stderr)
    sd.save_sample_aminer_random(data_path)
    sm.measures_sample_aminer_random(data_path)

    print("\nLoading matrices\n", file=sys.stderr)
    matrix = sl.load_matrix_jaccard_sim(data_path)
    print("Jaccard shape: %s\n" % str(matrix.shape), file=sys.stderr)
    matrix = sl.load_matrix_word2vec_sim(data_path)
    print("Word2vec shape: %s \n" % str(matrix.shape), file=sys.stderr)
    matrix = sl.load_matrix_jaccard_sim(data_path, related_docs=False)
    print("Jaccard shape: %s\n" % str(matrix.shape), file=sys.stderr)
    matrix = sl.load_matrix_word2vec_sim(data_path, related_docs=False)
    print("Word2vec shape: %s \n" % str(matrix.shape), file=sys.stderr)

if __name__ == "__main__":
    main()
