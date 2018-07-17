""" config/paths
    Module to configure paths.
"""
SRC_PACKAGE_PATH =  'src/somhos'
RESOURCES_PATH = SRC_PACKAGE_PATH + '/resources'
V9_PATH = RESOURCES_PATH + '/aminer/v9'
V9BETA_PATH = RESOURCES_PATH + '/aminer/v9beta'
V9GAMMA_PATH = RESOURCES_PATH + '/aminer/v9gamma'
DEAULT_WORDVECTORS = RESOURCES_PATH + '/GoogleNews-vectors-negative300.bin'
AMINER_ACM_SUFFIX = "/acm.txt"
KPS_DIRECTORY_INVERSE_SUFFIX = "/kps-directory-inverse-simpseq10-nopost.pkl"
KPS_DIRECTORY_SUFFIX = "/kps-directory-simpseq10-nopost.pkl"
KPS_NORMALIZED_SUFFIX = "/kps-normalized-simpseq10-nopost.pkl"
# Sample doc ids, content, measures and matrices
SAMPLE_PATH = "/samples"
DOCS_SAMPLE_A_SUFFIX = SAMPLE_PATH + "/docs-sample-a.pkl"
DOCS_SAMPLE_B_SUFFIX = SAMPLE_PATH + "/docs-sample-b.pkl"
DOCS_SAMPLES_CONTENT = SAMPLE_PATH + "/docs-samples-content.pkl"
DOCS_SAMPLES_WORD_COUNT = SAMPLE_PATH + "/docs-samples-word-count.pkl"
DOCS_SAMPLES_WORD_DOC_COUNT = SAMPLE_PATH + "/docs-samples-word-doc-count.pkl"
DOCS_SAMPLES_KPS_COUNT = SAMPLE_PATH + "/docs-samples-kps-count.pkl"
DOCS_SAMPLES_KPS_DOC_COUNT = SAMPLE_PATH + "/docs-samples-kps-doc-count.pkl"
# Measures
# Jaccard
DOCS_SAMPLES_JACCARD_SIM = SAMPLE_PATH + "/docs-samples-jaccard-sim.pkl"
DOCS_SAMPLES_JACCARD_SIM_UDV = SAMPLE_PATH + "/docs-samples-jaccard-sim-udv.pkl"
DOCS_SAMPLES_JACCARD_SIM_KPS = SAMPLE_PATH + "/docs-samples-jaccard-sim-kps.pkl"
DOCS_SAMPLES_JACCARD_SIM_UDV_KPS = SAMPLE_PATH + "/docs-samples-jaccard-sim-udv-kps.pkl"
# word2vec
DOCS_SAMPLES_WORD2VEC_SIM = SAMPLE_PATH + "/docs-samples-word2vec-sim.pkl"
DOCS_SAMPLES_WORD2VEC_SIM_UDV = SAMPLE_PATH + "/docs-samples-word2vec-sim-udv.pkl"
DOCS_SAMPLES_WORD2VEC_SIM_KPS = SAMPLE_PATH + "/docs-samples-word2vec-sim-kps.pkl"
DOCS_SAMPLES_WORD2VEC_SIM_UDV_KPS = SAMPLE_PATH + "/docs-samples-word2vec-sim-udv-kps.pkl"

# TF-IDF
DOCS_SAMPLES_TF_IDF = SAMPLE_PATH + "/docs-samples-tf-idf.pkl"
DOCS_SAMPLES_TF_IDF_UDV = SAMPLE_PATH + "/docs-samples-tf-idf-udv.pkl"
DOCS_SAMPLES_TF_IDF_KPS = SAMPLE_PATH + "/docs-samples-tf-idf-kps.pkl"
DOCS_SAMPLES_TF_IDF_UDV_KPS = SAMPLE_PATH + "/docs-samples-tf-idf-udv-kps.pkl"

# word2vec * Itf-idf
DOCS_SAMPLES_WORD2VEC_TFIDF = SAMPLE_PATH + "/docs-samples-word2vec-tfidf.pkl"
DOCS_SAMPLES_WORD2VEC_TFIDF_UDV = SAMPLE_PATH + "/docs-samples-word2vec-tfidf-udv.pkl"
DOCS_SAMPLES_WORD2VEC_TFIDF_KPS = SAMPLE_PATH + "/docs-samples-word2vec-tfidf-kps.pkl"
DOCS_SAMPLES_WORD2VEC_TFIDF_UDV_KPS = SAMPLE_PATH + "/docs-samples-word2vec-tfidf-udv-kps.pkl"

def get_relative_path(data_path, extra_path):
    """Merge path"""
    return data_path + extra_path
