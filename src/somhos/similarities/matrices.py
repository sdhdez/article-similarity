""" Methods to measure similarity matrices """
import os.path
import pickle

import numpy as np
from scipy import spatial
import gensim

from somhos.resources import dataset as rd
import somhos.methods.useful as mu
from somhos.methods.useful import print_log

DEAULT_WORDVECTORS = './src/somhos/resources/GoogleNews-vectors-negative300.bin'
MEASURES_PATH = '/measures'
DOCS_CENTROIDS_PATH = '/docs-centroids'

def measures_sample_aminer(data_path, docs, extra_suffix=""):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    # Form measures path
    measures_path = get_measures_path(data_path, extra_suffix=extra_suffix)
    if docs:
        docs_ids = docs['docs_ids']
        documents = docs['docs']
        # Get number of documents
        len_docs_ids = len(docs_ids)
        # Load measures if file exists
        measures = {'jaccard': load_measures(measures_path, "jaccard"),
                    'word2vec': load_measures(measures_path, "word2vec")}
        # If the expected number of measures doesn't match then
        # calculate the measures
        expected_measures = len_docs_ids*(len_docs_ids+1)//2
        if len(measures['jaccard']) != expected_measures:
            # Clear previous measures
            del measures
            measures = {'jaccard': {},
                        'word2vec': {}}
            print_log(" - Re-measuring similarities, %s expected" % expected_measures)
            print_log("Loading centroids ...")
            docs_centroids = get_docs_centroids(documents)
            # measures_batch = {'word2vec': {}}
            print_log("Measuring similarities ...")
            # Iter over document ids (triangular matrix)
            counter = 0
            for d_i, doc_id in enumerate(docs_ids):
                # Row document
                doc_i = documents[doc_id]
                # j = i to avoid repetition
                d_j = d_i
                # Columns
                while d_j < len_docs_ids:
                    # Col document
                    doc_j = documents[docs_ids[d_j]]
                    # Unique id for pair of documents
                    measure_id = doc_id + docs_ids[d_j]
                    # Get jaccard similarity
                    measures['jaccard'][measure_id] = get_jaccard_sim(doc_i, doc_j)
                    # Get word2vec similarity
                    measures['word2vec'][measure_id] = get_word2vec_centroid_sim(
                        doc_id, docs_ids[d_j], docs_centroids)
                    # Change column
                    d_j += 1
                    counter += 1
                    print_log(" - Document pair: %d, %d" % (d_i, d_j), cond=(counter % 50000 == 0))
            # Save measures
            # update_measures_batch(measures, measures_batch, 'word2vec')
            save_measures(measures_path, measures)
    else:
        print_log("Index doesn't exists")

def update_measures_batch(measures, measures_batch, method):
    """Update dict with results from batch of tensors"""
    if measures_batch[method]:
        measures[method].update(mu.tensor_to_value(measures_batch[method]))
        del measures_batch[method]
        measures_batch[method] = {}

def get_measures_path(data_path, extra_suffix=""):
    """Receive path to resources and return default path to save measures"""
    # Get suffix for current sample parameters
    suffix_path = rd.get_default_suffix(extra_suffix=extra_suffix)
    return data_path + MEASURES_PATH + suffix_path

def get_docs_centroids_path(data_path, extra_suffix=""):
    """Receive path to resources and return default path to save docs's centroids"""
    # Get suffix for current sample parameters
    suffix_path = rd.get_default_suffix(extra_suffix=extra_suffix)
    return data_path + DOCS_CENTROIDS_PATH + suffix_path + ".bin"

def measures_sample_aminer_related(data_path):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    # Load docs and ids in memory
    docs = rd.get_sample_aminer_related(data_path)
    extra_suffix = rd.get_default_extra_suffix(related_docs=True)
    measures_sample_aminer(data_path, docs, extra_suffix=extra_suffix)

def measures_sample_aminer_random(data_path):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    # Load docs and ids in memory
    docs = rd.get_sample_aminer_random(data_path)
    extra_suffix = rd.get_default_extra_suffix(related_docs=False)
    measures_sample_aminer(data_path, docs, extra_suffix=extra_suffix)

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

def get_docs_centroids(documents):
    """Receive dict with documents and word2vec model and return it of tensors"""
    docs_centroids = {}
    print_log("Loading word2vec model ...")
    wordvectors = gensim.models.KeyedVectors.load_word2vec_format(DEAULT_WORDVECTORS,
                                                                  binary=True)
    for doc_id, doc in documents.items():
        docs_centroids[doc_id] = mu.wordvectors_centroid(wordvectors, doc['bag_of_words'])
    # Free memory
    print_log("Unloading word2vec model ...")
    del wordvectors
    return docs_centroids

def get_word2vec_centroid_sim(id_a, id_b, docs_centroids):
    """Receive two doc ids and return tensor with vector similarity"""
    sim = 1 - spatial.distance.cosine(docs_centroids[id_a], docs_centroids[id_b])
    # Return similarity
    return sim

def load_merge_pickles(path_to_pickles, file_preffix="%d"):
    """Load and merge pickle files in a dict"""
    if os.path.exists(path_to_pickles):
        print_log("Loading pickles ...")
        pickle_number = 0
        content = {}
        el_sum = 0
        while True:
            path_to_pickle_file = path_to_pickles + file_preffix % pickle_number
            if os.path.exists(path_to_pickle_file):
                with open(path_to_pickle_file, "rb") as fin:
                    partal_content = pickle.load(fin)
                    el_sum += len(partal_content)
                    content.update(partal_content)
                    del partal_content
                pickle_number += 1
            else:
                break
        print_log(" - Loaded measures %s = %s" % (el_sum, len(content)))
        return content

def save_dict_pickles(path_to_pickles, content, file_preffix="%d"):
    """Receive dictionary an save it into pickle files"""
    len_dic = len(content)
    print_log(" - Saving %d dict" % len_dic)
    if not os.path.exists(path_to_pickles):
        os.mkdir(path_to_pickles)
    partial_content = {}
    i = 1
    pickle_number = 0
    for key, value in content.items():
        partial_content[key] = value
        if i % 2000000 == 0:
            print_log(" - %d elements" % i)
            path_to_pickle_file = path_to_pickles + file_preffix % pickle_number
            save_dict_pickle(path_to_pickle_file, partial_content)
            pickle_number += 1
            del partial_content
            partial_content = {}
        i += 1
    path_to_pickle_file = path_to_pickles + file_preffix % pickle_number
    save_dict_pickle(path_to_pickle_file, partial_content)

def save_dict_pickle(path_to_pickle_file, content):
    """Receive filename and save content to pikle file"""
    with open(path_to_pickle_file, "wb") as fout:
        pickle.dump(content, fout)

def save_measures(measures_path, measures):
    """Receive a path resource and a dictionary with pre-computed measures"""
    for method in measures:
        measures_method_path = measures_path + "-" + method + ".bin"
        # Open file to save measures
        with open(measures_method_path, "wb") as fout:
            print_log("Saving %d measures to file ...\
                      \n - File: %s " % (len(measures[method]), measures_method_path))
            # Save measures
            pickle.dump(measures[method], fout)
    return True

def load_measures(measures_path, method):
    """Receive path to resource and load saved measures"""
    measures_method_path = measures_path + "-" + method + ".bin"
    print_log("Path: %s" %  measures_method_path)
    measures = {}
    if os.path.exists(measures_method_path):
        with open(measures_method_path, "rb") as fin:
            print_log("Loading measures from file ...")
            # Load measures
            measures = pickle.load(fin)
            print_log(" - %d measures loaded from %s " % \
                      (len(measures), measures_method_path))
    return measures
