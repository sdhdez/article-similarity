import sys
from resources import dataset as rd 
from similarities import matrices as sm

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
    data_path = sys.argv[1] if sys.argv[1:2] else "./resources/test"
    print("Generating indexes and matrices ...", file=sys.stderr)
    print("\nRelated documents", file=sys.stderr)
    rd.index_aminer_txt_in(data_path)
    rd.save_sample_aminer_related(data_path)
    sm.measures_sample_aminer_related(data_path)
    print("\nRandom documents", file=sys.stderr)
    rd.save_sample_aminer_random(data_path)
    sm.measures_sample_aminer_random(data_path)
