""" Methods to measure similarity matrices """
import sys
import os.path
import itertools
import pickle
import numpy as np
import gensim

from resources import dataset as rd

DEAULT_WORDVECTORS = './resources/GoogleNews-vectors-negative300.bin'
WORD2VEC_MEASURED_SIMILARITIES_CACHE = './resources/word2vec-measured-similarities-cache'
WORD2VEC_MEASURED_SIMILARITIES_PICKLE = '/measure-%s.pkl'
MEASURES_PATH = '/measures'

def measures_sample_aminer(data_path, docs, extra_suffix=""):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    index_data_path = data_path + rd.INDEX_DATA
    # Form measures path
    measures_path = get_measures_path(data_path, extra_suffix=extra_suffix)
    if os.path.exists(index_data_path):
        docs_ids = docs[0]
        docs = docs[1]
        # Get number of documents
        len_docs_ids = len(docs_ids)
        # Load measures if file exists
        measures = load_measures(measures_path)
        # If the expected number of measures doesn't match then
        # calculate the measures
        expected_measures = len_docs_ids*(len_docs_ids+1)//2
        if len(measures) != expected_measures:
            # Clear previous measures
            measures = {}
            print(" - Wrong measures, %s expected" % expected_measures, file=sys.stderr)
            # Load previous word2vec similarities
            sim_measures = load_wordvector_similarities()
            print("Loading word2vec model ...")
            word_vectors = gensim.models.KeyedVectors.load_word2vec_format(DEAULT_WORDVECTORS,
                                                                           binary=True)
            print("Starting to measure ...")
            # Iter over document ids (triangular matrix)
            for d_i, doc_id in enumerate(docs_ids):
                # Row document
                doc_i = docs[doc_id]
                # j = i to avoid repetition
                d_j = d_i
                # Columns
                while d_j < len_docs_ids:
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
            save_measures(measures_path, measures)
            # Save measured word2vec similarities
            save_wordvector_similarities(sim_measures)
    else:
        print("Index doesn't exists", file=sys.stderr)

def get_measures_path(data_path, extra_suffix=""):
    """Receive path to resources and return default path to save measures"""
    # Get suffix for current sample parameters
    suffix_path = rd.get_default_suffix(extra_suffix=extra_suffix)
    return data_path + MEASURES_PATH + suffix_path + ".bin"

def measures_sample_aminer_related(data_path):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    # Load docs and ids in memory
    docs = rd.get_sample_aminer_related(data_path)
    measures_sample_aminer(data_path, docs, extra_suffix="-related")

def measures_sample_aminer_random(data_path):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    # Load docs and ids in memory
    docs = rd.get_sample_aminer_random(data_path)
    measures_sample_aminer(data_path, docs)

def get_jaccard_sim(wordlist_a, wordlist_b):
    """Receive info for two documents and return jaccard similarity"""
    # Get info from documents
    cardinality_a = wordlist_a['cardinality']
    cardinality_b = wordlist_b['cardinality']
    # Bag of words
    bag_of_words_a = wordlist_a['bag_of_words']
    bag_of_words_b = wordlist_b['bag_of_words']
    # Cardinality of shared words
    cardinality_ab = len(bag_of_words_a & bag_of_words_b)
    # Jaccard similarity
    jaccard_sim = np.divide(cardinality_ab, (cardinality_a + cardinality_b - cardinality_ab))
    return jaccard_sim

def get_word2vec_sim(wordlist_a, wordlist_b, word_vectors, sim_measures):
    """Receive info for two documents, the word2vec model
    and a dictionary of precalculated similarities,
    Return word2vec similarity and save the similarity in the dctionary"""
    # Document's info
    bag_of_words_a = wordlist_a['bag_of_words']
    bag_of_words_b = wordlist_b['bag_of_words']
    # Initialize variables
    mean_sim = 0.0
    sim_sum = 0.0
    n_pairs = 0
    # Get all the posible pairs of words from both documents AxB
    for pair in itertools.product(bag_of_words_a, bag_of_words_b):
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
        except KeyError:
            # If fails similarity is 0.0
            sim = 0.0
        # Sum of similarities
        sim_sum = np.sum([sim_sum, sim])
        # Number of pairs |A|x|B|
        n_pairs += 1
        # Mean
        mean_sim = np.divide(sim_sum, n_pairs)
    # Return similarity
    return mean_sim

def load_wordvector_similarities():
    """Load word-pairs similarities from pickle files"""
    if os.path.exists(WORD2VEC_MEASURED_SIMILARITIES_CACHE):
        print("Opening cached similarity measures ...", file=sys.stderr)
        pickle_number = 0
        sim_measures = {}
        el_sum = 0
        while True:
            path_measure = WORD2VEC_MEASURED_SIMILARITIES_CACHE + \
                           WORD2VEC_MEASURED_SIMILARITIES_PICKLE % pickle_number
            if os.path.exists(path_measure):
                with open(path_measure, "rb") as fin:
                    tmp = pickle.load(fin)
                    fin.close()
                    el_sum += len(tmp)
                    sim_measures.update(tmp)
                    del tmp
                pickle_number += 1
            else:
                break
        print(" - Loaded measures %s = %s" % (el_sum, len(sim_measures)), file=sys.stderr)
        return sim_measures

def save_wordvector_similarities(sim_measures):
    """Receive dictironary of word-pairs similarities and save into pickle files"""
    len_dic = len(sim_measures)
    print(" - Saving %d measures" % len_dic)
    if not os.path.exists(WORD2VEC_MEASURED_SIMILARITIES_CACHE):
        os.mkdir(WORD2VEC_MEASURED_SIMILARITIES_CACHE)
    tmp_sim_measures = {}
    i = 1
    pickle_number = 0
    for key, sim in sim_measures.items():
        if sim < 1.0 and sim > 0.0:
            tmp_sim_measures[key] = sim
        if i % 2000000 == 0:
            print(" -", i, "pairs")
            save_part_pickle(pickle_number, tmp_sim_measures)
            pickle_number += 1
            del tmp_sim_measures
            tmp_sim_measures = {}
        i += 1
    save_part_pickle(pickle_number, tmp_sim_measures)

def save_part_pickle(pickle_number, tmp_sim_measures):
    """Receive filename and save dictionary of word-pairs similarities"""
    with open(WORD2VEC_MEASURED_SIMILARITIES_CACHE + \
              WORD2VEC_MEASURED_SIMILARITIES_PICKLE % pickle_number, "wb") as fout:
        pickle.dump(tmp_sim_measures, fout)

def save_measures(measures_path, measures):
    """Receive a path resource and a dictionary with pre-computed measures"""
    # Open file to save measures
    with open(measures_path, "wb") as fout:
        print("Saving %d measures to file ...\
                \n - File: %s " % (len(measures), measures_path), file=sys.stderr)
        # Save measures
        pickle.dump(measures, fout)
        fout.close()
        return True

def load_measures(measures_path):
    """Receive path to resource and load saved measures"""
    if os.path.exists(measures_path):
        with open(measures_path, "rb") as fin:
            print("Loading measures from file ...", file=sys.stderr)
            # Load measures
            measures = pickle.load(fin)
            fin.close()
            print(" - %d measures loaded from %s " % \
                    (len(measures), measures_path), file=sys.stderr)
            return measures
    else:
        return {}
