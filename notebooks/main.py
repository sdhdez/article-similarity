import sys
import resources.dataset as rd
import measures.jaccard as mjaccard

if __name__ == "__main__":
    print("Main from command-line.", file=sys.stderr)
    doc_list = []
    for pfilename in rd.get_filenames("./resources/aminer/v1"):
        for indexdoc, text in rd.get_target_text(pfilename):
            bag_of_words = rd.get_bag_of_words(rd.get_tokens_default(text.lower()))
            doc_list.append({ 'set': bag_of_words, 
                                'cardinality': len(bag_of_words),
                                'index': indexdoc
                                })
    print("Documents: ", len(doc_list))
    J = mjaccard.similarity_matrix(doc_list)
    print("J.shape ", J.shape)
    print(J[98,99])

