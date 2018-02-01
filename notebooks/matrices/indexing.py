import os.path
import numpy as np
from scipy.sparse import dok_matrix
import random
from whoosh.index import open_dir, create_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC
from whoosh import qparser
from whoosh.qparser import QueryParser
from whoosh.query import Term, And

INDEX_DATA = "/index-data"
INDEX_JACCARD = "/index-jaccard"

def index_jaccard(data_path, nrand=100.0):
    index_data_path = data_path + INDEX_DATA
    index_jaccard_path = data_path + INDEX_JACCARD + (("-" + str(nrand)) if nrand else "")

    if os.path.exists(index_data_path):
        print("Opening data ...")
        ix_data = open_dir(index_data_path)
        parser = QueryParser("bag_of_words", ix_data.schema, group=qparser.OrGroup)

        schema = Schema(setA=ID(stored=True), setB=ID(stored=True),
                        cA=NUMERIC(int, 32, signed=False, stored=True),
                        cB=NUMERIC(int, 32, signed=False, stored=True),
                        cAB=NUMERIC(int, 32, signed=False, stored=True),
                        sim=NUMERIC(float, stored=True),
                        dis=NUMERIC(float, stored=True)
                        )
        if not os.path.exists(index_jaccard_path):
            os.mkdir(index_jaccard_path)
        ix_jaccard = create_in(index_jaccard_path, schema)
        writer = ix_jaccard.writer(limitmb=2048)

        with ix_data.reader() as reader:
            with ix_data.searcher() as searcher:
                fixed_seed = reader.doc_count()
                roulette_threshold = (1.2*nrand)/fixed_seed
                random.seed(fixed_seed)
                print("Indexed documents: ", fixed_seed, roulette_threshold)
                i, docdict = 0, {} 
                for doc_i, doc in reader.iter_docs():
                    if i >= nrand:
                        break 
                    if random.random() > roulette_threshold:
                        continue
                    q = parser.parse(doc['bag_of_words_title'])
                    result = searcher.search(q, limit = None)
                    print(". ", doc["indexdoc"], len(result))

                    setA = set(doc['bag_of_words'].split())
                    cA = doc['cardinality']
                    for rdoc in result:
                        setB = set(rdoc['bag_of_words'].split())
                        cB = rdoc['cardinality']
                        cAB = float(len(setA & setB))
                        jaccard_sim = cAB/(cA + cB - cAB)
                        writer.add_document(setA=doc["indexdoc"], 
                                            setB=rdoc["indexdoc"],
                                            cA=cA, 
                                            cB=cB, 
                                            cAB=cAB, 
                                            sim=jaccard_sim, 
                                            dis=1.0-jaccard_sim)
                        docdict[rdoc["indexdoc"]] = 1
                    i += 1
                print("Random documents: ", i)
                print("Sample size: ", len(docdict))
        writer.commit()

def get_matrix(index_path):
    bag_of_words = []
    N = len(bag_of_words)
    dmatrix = dok_matrix((N, N), dtype=np.float32)
    i = 0
    while i < N:
        j = i
        if i != j and i % 1000 == 0:
            print("d(i) :", i)
        while j < N:
            jd = jaccard_index(bag_of_words[i], bag_of_words[j])
            if jd != 0.0:
                dmatrix[i,j] = jd 
            j += 1
        i += 1
    return dmatrix
