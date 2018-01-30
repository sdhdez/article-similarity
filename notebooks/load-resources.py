import sys
import resources.dataset as rd

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)

    generic_name_bow = "/bag_of_words_by_document.db"
    documents_path = sys.argv if sys.argv[1:2] else "./resources/test"
    bag_of_words_file = documents_path + generic_name_bow
    print("Reading documents ...")
    rd.create_bow_from_files_in(documents_path, bag_of_words_file)
