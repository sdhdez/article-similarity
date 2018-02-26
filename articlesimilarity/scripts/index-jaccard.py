import sys
import resources.dataset as rd
import matrices.indexing as mindexing

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
    data_path = sys.argv[1] if sys.argv[1:2] else "./resources/test"
    print("Indexing similarity matrix ...")
    mindexing.index_jaccard(data_path, data_path + "/index-data-sample-100-10")

