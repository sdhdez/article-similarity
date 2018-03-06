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
MEASURES_PATH = '/measures'

def measures_sample_aminer(data_path, measures_path, docs_ids, docs):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    index_data_path = data_path + rd.INDEX_DATA
    if os.path.exists(index_data_path):
        # Get number of documents
        docs_ids = docs_ids[:1]
        N = len(docs_ids)
        # Load measures if file exists 
        measures = load_measures(measures_path)
        # If the expected number of measures doesn't match then 
        # calculate the measures
        if len(measures) != N*(N+1)//2:
            # Clear previous measures 
            measures = {}
            print(" - Wrong measures ...", file=sys.stderr)
            # Load previous word2vec similarities 
            ix_data = load_index_wordvector_similarities()
            ix_word2vec_similarities_searcher, parser = load_searcher_wordvector_similarities(ix_data)
            ix_word2vec_similarities_writer = load_writer_wordvector_similarities(ix_data)
            print(" - Loading word2vec model ...")
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
                                                                        ix_word2vec_similarities_searcher, parser,
                                                                        ix_word2vec_similarities_writer)
                    # Change column 
                    d_j += 1
                    if d_j % 50000 == 0:
                        print(" - Document pair: %d" % d_j, file=sys.stderr)
            # Free memory
            print(" - Unloading word2vec model ...", file=sys.stderr)
            del word_vectors
            # Save measures 
            save_measures(data_path, measures_path, measures)
            # Save measured word2vec similarities
            ix_word2vec_similarities_writer.commit()
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

def get_word2vec_sim(A, B, word_vectors, ix_word2vec_similarities_searcher, parser, ix_word2vec_similarities_writer):
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
                sim = get_wordvector_similarity(pair_id, ix_word2vec_similarities_searcher, parser)
                if sim == None:
                    sim = word_vectors.similarity(pair[0], pair[1])
                    print(pair_id, sim)
                    save_wordvector_similarity(pair_id, sim, ix_word2vec_similarities_writer)
        except KeyError as e:
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

def load_index_wordvector_similarities():
    if os.path.exists(WORD2VEC_MEASURED_SIMILARITIES_CACHE):
        print("Opening indexed similarity measures ...", file=sys.stderr)
        # Open index
        ix_data = open_dir(WORD2VEC_MEASURED_SIMILARITIES_CACHE) 
        return ix_data

def load_searcher_wordvector_similarities(ix_data):
    """Load and return cached word2vec similarities
    Each item in the dictionary has a pair of words as key, the pair is ordered alphabetically.
    """
    # Parser for query terms 
    parser = QueryParser("pair", ix_data.schema)
    return ix_data.searcher(), parser

def load_writer_wordvector_similarities(ix_data):
    print("Creating writer to save similarity measures ...", file=sys.stderr)
    writer = ix_data.writer(limitmb=2048)
    return writer

def get_wordvector_similarity(pair, searcher, parser):
    # Searcher for all documents
    q = parser.parse(pair)
    result = searcher.search(q)
    for r in result:
        # print("Pair: sim = %s" % r["sim"])
        return r["sim"]

def save_wordvector_similarity(pair, sim, writer):
    # Define writer 
    writer.add_document(pair=pair, sim=sim)

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
            print(" -  %d measures loaded from %s " % (len(measures), measures_path), file=sys.stderr)
            return measures
    else:
        return {}
