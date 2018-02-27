import sys
from resources import dataset as rd 

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
    documents_path = sys.argv[1] if sys.argv[1:2] else "./resources/test"
    print("Reading documents ...")
    rd.index_aminer_txt_in(documents_path)
