import sys, os.path
import itertools
import pickle
import numpy as np
import gensim

from resources import dataset as rd 

DEAULT_WORDVECTORS = './resources/GoogleNews-vectors-negative300.bin'
WORD2VEC_MEASURED_SIMILARITIES_CACHE = './resources/word2vec-measured-similarities-cache.bin'
MEASURES_PATH = '/measures'

def measures_sample_aminer_related(data_path):
    """Receive a path to resources and save jaccard and word2vec similarities"""
    index_data_path = data_path + rd.INDEX_DATA
    if os.path.exists(index_data_path):
        # Load docs and ids in memory 
        docs_ids, docs = rd.get_sample_aminer_related(data_path)
        # Get number of documents
        N = len(docs_ids)
        # Save document similarities  
        measures = load_measures(data_path)
        if len(measures) != N*(N+1)//2:
            print("Loading model for word2vec ...")
            word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                                            DEAULT_WORDVECTORS, 
                                            binary=True
                                        )
            print("Generating measures ...", file=sys.stderr)
            # Load previous word2vec similarities 
            word2vec_similarities = load_cached_wordvector_similarities()
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
                    measures[measure_id]['word2vec'] = get_word2vec_sim(doc_i, doc_j, word_vectors, word2vec_similarities)
                    # Change column 
                    d_j += 1
            # Save measured word2vec similarities
            save_wordvector_similarities(word2vec_similarities)
            # Save measures 
            save_measures(data_path, measures)
    else:
        print("Index doesn't exists", file=sys.stderr)

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

def get_word2vec_sim(A, B, word_vectors, word2vec_similarities):
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
            # Order each pair by alphabetical order to normalize id of pairs  
            pair = sorted(pair)
            # Generate unique id (string)
            pair_id = ",".join(pair)
            # If pair exists in previous measures retrive the saved measure
            if pair_id in word2vec_similarities:
                sim = word2vec_similarities[pair_id]
            # Otherwise measure and save similarity 
            else:
                sim = word_vectors.similarity(pair[0], pair[1])
                word2vec_similarities[pair_id] = sim
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

def load_cached_wordvector_similarities():
    """Load and return cached word2vec  similarities
    Each item in the dictionary has a pair of words as key, the pair is ordered alphabetically.
    """
    # Open file if exists or return an empty dictionary
    if os.path.exists(WORD2VEC_MEASURED_SIMILARITIES_CACHE):
        with open(WORD2VEC_MEASURED_SIMILARITIES_CACHE, "rb") as fp:
            # Load dictionary 
            word2vec_similarities = pickle.load(fp)
            fp.close()
            # Return saved similarities
            return word2vec_similarities
    else:
        return {}

def save_wordvector_similarities(word2vec_similarities):
    """Receive a dictionary with measured similarities between pairs of words and 
    dump it in a file for future consulting."""
    with open(WORD2VEC_MEASURED_SIMILARITIES_CACHE, "wb") as fp:
        print("Saving %d word2vec measured similarities.\
                \n - File: %s " % (len(word2vec_similarities), WORD2VEC_MEASURED_SIMILARITIES_CACHE), file=sys.stderr)
        # Dump dictionary to a file 
        pickle.dump(word2vec_similarities, fp)
        fp.close()
        return True

def save_measures(data_path, measures):
    """Receive a path resource and a dictionary with pre-computed measures"""
    # Get suffix for current sample parameters
    suffix_path = rd.get_default_suffix()
    # Form measures path 
    measures_path = data_path + MEASURES_PATH + suffix_path + ".bin"
    # Open file to save measures
    with open(measures_path, "wb") as fp:
        print("Saving %d measures to file ...\
                \n - File: %s " % (len(measures), measures_path), file=sys.stderr)
        # Save measures
        pickle.dump(measures, fp)
        fp.close()
        return True

def load_measures(data_path):
    """Receive path to resource and load saved measures"""
    # Get suffix for current sample parameters
    suffix_path = rd.get_default_suffix()
    # Form measures path 
    measures_path = data_path + MEASURES_PATH + suffix_path + ".bin"
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

def index_word2vec(data_path, index_data_sample_path):
    model = gensim.models.KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True)

    index_word2vec_path = data_path + INDEX_WORD2VEC

    if os.path.exists(index_data_sample_path):
        print("Opening data samples ...")
        ix_data_sample = open_dir(index_data_sample_path)

        schema = Schema(setA=ID(stored=True), setB=ID(stored=True),
                        sim=NUMERIC(float, stored=True),
                        dis=NUMERIC(float, stored=True),
                        n_pairs=NUMERIC(int, stored=True) #,
                        # pairs_scores=STORED(),
                        # bag_of_words_A=STORED(),
                        # bag_of_words_B=STORED()
                        )
        if not os.path.exists(index_word2vec_path):
            os.mkdir(index_word2vec_path)
        ix_word2vec = create_in(index_word2vec_path, schema)
        writer = ix_word2vec.writer(limitmb=2048)

        with ix_data_sample.reader() as rows:
            print("Indexed samples: ", rows.doc_count())
            with ix_data_sample.reader() as cols:
                for doc_i, doc in rows.iter_docs():
                    setA = doc['bag_of_words'].split()
                    for rdoc_i, rdoc in cols.iter_docs():
                        setB = rdoc['bag_of_words'].split()
                        pairs = []
                        for w1, w2 in itertools.product(setA,setB):
                            try:
                                simw1w2 = model.wv.similarity(w1, w2)
                                pairs.append(((w1 + "," + w2 + "="), simw1w2))
                            except KeyError as e:
                                pass
                        n_pairs = len(pairs)
                        mean_sim = (sum([f for s,f in pairs])/n_pairs) if n_pairs > 0.0 else 0.0
                        pairs_scores = " ".join([(s + str(f)) for s,f in pairs])
                        if (doc_i + 1) % 50 == 0:
                            print(doc_i + 1, ". ", doc["indexdoc"], rdoc["indexdoc"], mean_sim)
                        word2vec_sim = mean_sim
                        if word2vec_sim > 0.0:
                            writer.add_document(setA=doc["indexdoc"], 
                                                setB=rdoc["indexdoc"],
                                                sim=word2vec_sim, 
                                                dis=1.0-word2vec_sim,
                                                n_pairs=n_pairs #,
                                                # pairs_scores=pairs_scores,
                                                # bag_of_words_A=doc["bag_of_words"],
                                                # bag_of_words_B=rdoc["bag_of_words"]
                                                )
                print("MxN: ", doc_i + 1, rdoc_i + 1, (doc_i + 1) * (rdoc_i + 1))
        print("Commiting ...")
        writer.commit()
        ix_word2vec = open_dir(index_word2vec_path)
        with ix_word2vec.reader() as reader:
            print("Indexed measures (word2vec): ", reader.doc_count())

def reindex_matrix_word2vec_sim(data_path):
    index_word2vec_path = data_path + INDEX_WORD2VEC
    index_word2vec_full_path = data_path + INDEX_WORD2VEC + '-full'
    if os.path.exists(index_word2vec_full_path):
        if not os.path.exists(index_word2vec_path):
            os.mkdir(index_word2vec_path)

        schema = Schema(setA=ID(stored=True), setB=ID(stored=True),
                        sim=NUMERIC(float, stored=True),
                        dis=NUMERIC(float, stored=True),
                        n_pairs=NUMERIC(int, stored=True),
                        #pairs_scores=STORED(),
                        #bag_of_words_A=STORED(),
                        #bag_of_words_B=STORED()
                        )

        ix_word2vec_full = open_dir(index_word2vec_full_path)
        ix_word2vec = create_in(index_word2vec_path, schema)
        writer = ix_word2vec.writer(limitmb=1024)

        with ix_word2vec_full.reader() as reader:
            for doc_i, doc in reader.iter_docs():
                writer.add_document(setA=doc["setA"], 
                                        setB=doc["setB"],
                                        sim=doc["sim"], 
                                        dis=doc["dis"],
                                        n_pairs=doc["n_pairs"]
                                        #pairs_scores=doc["pairs_scores"],
                                        #bag_of_words_A=doc["bag_of_words_A"],
                                        #bag_of_words_B=doc["bag_of_words_B"]
                                        )
        print("Commiting ...")
        writer.commit()

