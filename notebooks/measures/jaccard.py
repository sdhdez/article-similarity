import os.path
import numpy as np
from scipy.sparse import dok_matrix
from whoosh.index import open_dir, create_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC
from whoosh import qparser
from whoosh.qparser import QueryParser
from whoosh.query import Term, And

def jaccard_index(set_A, set_B):
    card_intersection_A_B = float(len(set_A['set'] & set_B['set']))
    return card_intersection_A_B /(set_A['cardinality'] + set_B['cardinality'] - card_intersection_A_B)

# def jaccard_distance(set_A, set_B):
#    return 1.0 - jaccard_index(set_A, set_B)

def test_whoosh(data_path):
    index_data_path = data_path + "/index-data"
    if os.path.exists(index_data_path):
        print("Opening data ...")
        ix = open_dir(index_data_path)
        with ix.searcher() as searcher:
            parser = QueryParser("bag_of_words", ix.schema, group=qparser.OrGroup)
            myquery = parser.parse(u'regression models them determine new network propose outperforms microbiological evolutionary data use benchmark been features based learning obtain between method sets linear applied hybridation afterwards good has classification product obtained covariates techniques model promising apply structure neural part space terms compared results seven complexity unit classifier well compromise very other accuracy obtaining algorithm first perform binary step nonlinear several logistic using problem they derived hybrid basic')
            print(myquery)
            result = searcher.search(myquery)
            print(len(result))
            for e in result:
                print(str(e['bag_of_words']).encode('utf8'))

def index_similarities(data_path):
    index_data_path = data_path + "/index-data"
    index_jaccard_path = data_path + "/index-jaccard"

    if os.path.exists(index_data_path):
        print("Opening data ...")
        data_batch = 10000
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
                for doc_i, doc in reader.iter_docs():
                    q = parser.parse(doc['bag_of_words_title'])
                    result = searcher.search(q)

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
                        
                    if (doc_i + 1) % data_batch == 0:
                        print("N: ", doc_i) 
        writer.commit()

def similarity_matrix(data_path):
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
