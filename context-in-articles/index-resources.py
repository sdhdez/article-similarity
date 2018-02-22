import sys
import resources.dataset as rd

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)

    documents_path = sys.argv[1] if sys.argv[1:2] else "./resources/test"
    print("Reading documents ...")
    rd.create_bow_from_files_in(documents_path)
