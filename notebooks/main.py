import sys
import resources.dataset as rd
import measures.jaccard as mjaccard

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
  
    doc_list = {}
    for pfilename in rd.get_filenames("./resources/test"):
        for indexdoc, text in rd.get_target_text(pfilename):
            bag_of_words = rd.get_bag_of_words(rd.get_tokens_default(text.lower()))
            doc_list[indexdoc] = { 'set': bag_of_words, 
                                    'cardinality': len(bag_of_words)}
    #print([doc_list[d]['cardinality'] for d in doc_list])
    mjaccard.distance_matrix(doc_list)

