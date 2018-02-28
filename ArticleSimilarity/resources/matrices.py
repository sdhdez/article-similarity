import os.path
import numpy as np
import random
import itertools
from scipy.sparse import dok_matrix
from whoosh.index import open_dir, create_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC, STORED
from whoosh import qparser
from whoosh.qparser import QueryParser
from whoosh.query import Term, And
import gensim

# Import default paths to indexed data

def index_jaccard(data_path, index_data_sample_path):
    index_jaccard_path = data_path + INDEX_JACCARD

    if os.path.exists(index_data_sample_path):
        print("Opening sample data ...")
        ix_data_sample = open_dir(index_data_sample_path)

        schema = Schema(setA=ID(stored=True), setB=ID(stored=True),
                        cA=NUMERIC(int, 32, signed=False, stored=True),
                        cB=NUMERIC(int, 32, signed=False, stored=True),
                        cAB=NUMERIC(int, 32, signed=False, stored=True),
                        sim=NUMERIC(float, stored=True),
                        dis=NUMERIC(float, stored=True),
                        bag_of_words_A=STORED(),
                        bag_of_words_B=STORED()
                        )
        if not os.path.exists(index_jaccard_path):
            os.mkdir(index_jaccard_path)
        ix_jaccard = create_in(index_jaccard_path, schema)
        writer = ix_jaccard.writer(limitmb=2048)

        with ix_data_sample.reader() as rows:
            print("Indexed samples: ", rows.doc_count())
            with ix_data_sample.reader() as cols:
                for doc_i, doc in rows.iter_docs():
                    setA = set(doc['bag_of_words'].split())
                    cA = doc['cardinality']
                    for rdoc_i, rdoc in cols.iter_docs():
                        print(". ", doc["indexdoc"], rdoc["indexdoc"])
                        setB = set(rdoc['bag_of_words'].split())
                        cB = rdoc['cardinality']
                        cAB = float(len(setA & setB))
                        if cAB > 0.0:
                            jaccard_sim = cAB/(cA + cB - cAB)
                            writer.add_document(setA=doc["indexdoc"], 
                                            setB=rdoc["indexdoc"],
                                            cA=cA, 
                                            cB=cB, 
                                            cAB=cAB, 
                                            sim=jaccard_sim, 
                                            dis=1.0-jaccard_sim,
                                            bag_of_words_A=doc["bag_of_words"],
                                            bag_of_words_B=rdoc["bag_of_words"]
                                            )
                    print("MxN: ", doc_i, rdoc_i, doc_i * rdoc_i)
        writer.commit()
        ix_jaccard = open_dir(index_jaccard_path)
        with ix_jaccard.reader() as reader:
            print("Indexed measures: ", reader.doc_count())

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

