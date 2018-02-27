import sys
# from ArticleSimilarity.resources import dataset # create_bow_from_files_in 

print(__name__)

print("Main from command-line.", file=sys.stderr)
documents_path = sys.argv[1] if sys.argv[1:2] else "../resources/test"
print("Reading documents ...")
create_bow_from_files_in(documents_path)
