import sys, os.path
import itertools
import pickle
import numpy as np
import gensim

from resources import dataset as rd 

from whoosh.index import open_dir 
from whoosh.fields import Schema, ID, STORED
from whoosh.qparser import QueryParser

DEAULT_WORDVECTORS = './resources/GoogleNews-vectors-negative300.bin'
WORD2VEC_MEASURED_SIMILARITIES_CACHE = './resources/word2vec-measured-similarities-cache'
WORD2VEC_MEASURED_SIMILARITIES_PICKLE = '/measure-%s.pkl'
MEASURES_PATH = '/measures'

def measures_sample_aminer(data_path, measures_path, docs_ids, docs):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    index_data_path = data_path + rd.INDEX_DATA
    if os.path.exists(index_data_path):
        # Get number of documents
        N = len(docs_ids)
        # Load measures if file exists 
        measures = load_measures(measures_path)
        # If the expected number of measures doesn't match then 
        # calculate the measures
        expected_measures = N*(N+1)//2
        if len(measures) != expected_measures:
            # Clear previous measures 
            measures = {}
            print(" - Wrong measures, %s expected" % expected_measures, file=sys.stderr)
            # Load previous word2vec similarities 
            sim_measures = load_dict_wordvector_similarities()
            print("Loading word2vec model ...")
            word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                                            DEAULT_WORDVECTORS, 
                                            binary=True)
            # Iter over document ids (triangular matrix)
            for d_i, doc_id in enumerate(docs_ids):
                # Row document
                doc_i = docs[doc_id]
                # j = i to avoid repetition 
                d_j = d_i
                # Columns 
                while d_j < N:
                    # Col document
                    doc_j = docs[docs_ids[d_j]]
                    # Unique id for pair of documents 
                    measure_id = doc_id + docs_ids[d_j]
                    # Initialize dictionary with the id
                    measures.setdefault(measure_id, {'word2vec': 0.0, 'jaccard': 0.0})
                    # Get jaccard similarity
                    measures[measure_id]['jaccard'] = get_jaccard_sim(doc_i, doc_j)
                    # Get word2vec similarity
                    measures[measure_id]['word2vec'] = get_word2vec_sim(doc_i, doc_j, 
                                                                        word_vectors, 
                                                                        sim_measures)
                    # Change column 
                    d_j += 1
                    if d_j % 50000 == 0:
                        print(" - Document pair: %d" % d_j, file=sys.stderr)
            # Free memory
            print("Unloading word2vec model ...", file=sys.stderr)
            del word_vectors
            # Save measures 
            save_measures(data_path, measures_path, measures)
            # Save measured word2vec similarities
            save_dict_wordvector_similarities(sim_measures)
    else:
        print("Index doesn't exists", file=sys.stderr)

def measures_sample_aminer_related(data_path):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    # Get suffix for current sample parameters
    suffix_path = rd.get_default_suffix(extra_suffix="-related")
    # Form measures path 
    measures_path = data_path + MEASURES_PATH + suffix_path + ".bin"
    # Load docs and ids in memory 
    docs_ids, docs = rd.get_sample_aminer_related(data_path)
    measures_sample_aminer(data_path, measures_path, docs_ids, docs)

def measures_sample_aminer_random(data_path):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    # Get suffix for current sample parameters
    suffix_path = rd.get_default_suffix()
    # Form measures path 
    measures_path = data_path + MEASURES_PATH + suffix_path + ".bin"
    # Load docs and ids in memory 
    docs_ids, docs = rd.get_sample_aminer_random(data_path)
    measures_sample_aminer(data_path, measures_path, docs_ids, docs)

def get_jaccard_sim(A, B):
    """Receive info for two documents and return jaccard similarity"""
    # Get info from documents 
    cA = A['cardinality']
    cB = B['cardinality']
    # Bag of words 
    setA = A['bag_of_words']
    setB = B['bag_of_words']
    # Cardinality of shared words
    cAB = len(setA & setB)
    # Jaccard similarity
    jaccard_sim = np.divide(cAB, (cA + cB - cAB))
    return jaccard_sim

def get_word2vec_sim(A, B, word_vectors, sim_measures):
    """Receive info for two documents, the word2vec model and a dictionary of precalculated similarities, 
    Return word2vec similarity and save the similarity in the dctionary"""
    # Document's info
    setA = A['bag_of_words']
    setB = B['bag_of_words']
    # Initialize variables
    mean_sim = 0.0
    sim_sum = 0.0
    n_pairs = 0
    # Get all the posible pairs of words from both documents AxB
    for pair in itertools.product(setA,setB):
        try:
            if pair[0] == pair[1]:
                sim = 1.0
            else:
                # Order each pair by alphabetical order to normalize id of pairs  
                pair = sorted(pair)
                # Generate unique id (string)
                pair_id = ",".join(pair)
                # If pair exists in previous measures retrive the saved measure
                if pair_id in sim_measures:
                    sim = sim_measures[pair_id]
                else:
                    sim = word_vectors.similarity(pair[0], pair[1])
                    sim_measures[pair_id] = sim 
        except KeyError as e:
            # If fails similarity is 0.0
            sim = 0.0
            sim_measures[pair_id] = sim 
        # Sum of similarities 
        sim_sum = np.sum([sim_sum, sim])
        # Number of pairs |A|x|B|
        n_pairs += 1
        # Mean 
        mean_sim = np.divide(sim_sum, n_pairs)
    # Return similarity
    return mean_sim

def load_dict_wordvector_similarities():
    if os.path.exists(WORD2VEC_MEASURED_SIMILARITIES_CACHE):
        print("Opening cached similarity measures ...", file=sys.stderr)
        p = 0
        sim_measures = {}
        el_sum = 0
        while True:
            path_measure = WORD2VEC_MEASURED_SIMILARITIES_CACHE + WORD2VEC_MEASURED_SIMILARITIES_PICKLE % p
            if os.path.exists(path_measure):
                with open(path_measure, "rb") as fin:
                    tmp = pickle.load(fin)
                    fin.close()
                    el_sum += len(tmp)
                    sim_measures.update(tmp)
                    del tmp
                p+=1
            else:
                break
        print(" - Loaded measures %s = %s" % (el_sum, len(sim_measures)), file=sys.stderr)
        return sim_measures

def save_dict_wordvector_similarities(sim_measures):
    len_dic = len(sim_measures)
    print(" - Saving %d measures" % len_dic)
    if not os.path.exists(WORD2VEC_MEASURED_SIMILARITIES_CACHE):
        os.mkdir(WORD2VEC_MEASURED_SIMILARITIES_CACHE)
    tmp_sim_measures = {}
    i = 1
    p = 0
    for k,v in sim_measures.items():
        if v < 1.0:
            tmp_sim_measures[k] = v
        if i % 5000000 == 0:
            print(" -", i, "pairs")
            save_part_pickle(p, tmp_sim_measures)
            p+=1
            del tmp_sim_measures
            tmp_sim_measures = {}
        i+=1
    save_part_pickle(p, tmp_sim_measures)

def save_part_pickle(p, tmp_sim_measures):
    with open(WORD2VEC_MEASURED_SIMILARITIES_CACHE + WORD2VEC_MEASURED_SIMILARITIES_PICKLE % p, "wb") as fout:
        pickle.dump(tmp_sim_measures, fout)
        fout.close()

def save_measures(data_path, measures_path, measures):
    """Receive a path resource and a dictionary with pre-computed measures"""
    # Open file to save measures
    with open(measures_path, "wb") as fp:
        print("Saving %d measures to file ...\
                \n - File: %s " % (len(measures), measures_path), file=sys.stderr)
        # Save measures
        pickle.dump(measures, fp)
        fp.close()
        return True

def load_measures(measures_path):
    """Receive path to resource and load saved measures"""
    if os.path.exists(measures_path):
        with open(measures_path, "rb") as fp:
            print("Loading measures from file ...", file=sys.stderr)
            # Load measures
            measures = pickle.load(fp)
            fp.close()
            print(" - %d measures loaded from %s " % (len(measures), measures_path), file=sys.stderr)
            return measures
    else:
        return {}
