import sys
from resources import dataset as rd 
from similarities import matrices as sm

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
    data_path = sys.argv[1] if sys.argv[1:2] else "./resources/test"
    print("Reading documents ...")
    rd.index_aminer_txt_in(data_path)
    rd.save_sample_aminer_related(data_path)
    sm.measures_sample_aminer_related(data_path)
