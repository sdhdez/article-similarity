import sys, os.path
import itertools
import numpy as np
import gensim

from resources import dataset as rd 

def measures_sample_aminer_related(data_path):
    """Receive a path to an index and save jaccard and word2vec similarities"""
    index_data_path = data_path + rd.INDEX_DATA
    if os.path.exists(index_data_path):
        print("Loading model for word2vec ...")
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                                            './resources/GoogleNews-vectors-negative300.bin', 
                                            binary=True
                                        )
        print("Generating measures ...", file=sys.stderr)
        # Load docs and ids in memory 
        docs_ids, docs = rd.get_sample_aminer_related(data_path)
        docs_ids = docs_ids[:2]
        # Get number of documents
        N = len(docs_ids)
        # Save document similarities  
        measures = {}
        # Save word2vec similarities 
        word2vec_similarities = {}
        # Iter over docment ids (triangular matrix)
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
        print(word2vec_similarities, len(word2vec_similarities))
        print(len(measures), N*(N+1)//2)
    else:
        print("Index doesn't exists", file=sys.stderr)

def get_jaccard_sim(A, B):
    cA = A['cardinality']
    cB = B['cardinality']
    setA = A['bag_of_words']
    setB = B['bag_of_words']
    cAB = len(setA & setB)
    jaccard_sim = np.divide(cAB, (cA + cB - cAB))
    return jaccard_sim

def get_word2vec_sim(A, B, word_vectors, word2vec_similarities):
    setA = A['bag_of_words']
    setB = B['bag_of_words']
    pairs = []
    mean_sim = 0.0
    sim_sum = 0.0
    n_pairs = 0.0
    for pair in itertools.product(setA,setB):
        try:
            pair = sorted(pair)
            pair_id = ",".join(pair)
            if pair_id in word2vec_similarities:
                sim += word2vec_similarities[pair_id]
            else:
                sim = word_vectors.similarity(pair[0], pair[1])
                word2vec_similarities[pair_id] = sim
        except KeyError as e:
            sim = 0.0
        sim_sum += sim
        n_pairs += 1
        mean_sim = np.divide(sim_sum, n_pairs)
    return mean_sim

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

