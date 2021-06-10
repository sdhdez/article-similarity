""" config/paths
    Module to configure paths.
"""
SRC_PACKAGE_PATH = 'src/somhos'
RESOURCES_PATH = SRC_PACKAGE_PATH + '/resources'
V9_PATH = RESOURCES_PATH + '/aminer/v9'
V9BETA_PATH = RESOURCES_PATH + '/aminer/v9beta'
V9GAMMA_PATH = RESOURCES_PATH + '/aminer/v9gamma'
DEAULT_WORDVECTORS = RESOURCES_PATH + '/GoogleNews-vectors-negative300.bin'
AMINER_ACM_SUFFIX = "/acm.txt"
KPS_DIRECTORY_INVERSE_SUFFIX = "/kps-directory-inverse-simpseq10-nopost.pkl"
KPS_DIRECTORY_SUFFIX = "/kps-directory-simpseq10-nopost.pkl"
KPS_NORMALIZED_SUFFIX = "/kps-normalized-simpseq10-nopost.pkl"
KPS_DOCS_IDS_SUFFIX = "/kps-docs-ids-simpseq10-nopost.pkl"
# Sample doc ids, content, measures and matrices
SAMPLE_PATH = "/samples"
DOCS_SAMPLE_A_SUFFIX = SAMPLE_PATH + "/docs-sample-a.pkl"
DOCS_SAMPLE_B_SUFFIX = SAMPLE_PATH + "/docs-sample-b.pkl"
DOCS_SAMPLES_CONTENT = SAMPLE_PATH + "/docs-samples-content.pkl"
DOCS_SAMPLES_WORD_COUNT = SAMPLE_PATH + "/docs-samples-word-count.pkl"
DOCS_SAMPLES_WORD_DOC_COUNT = SAMPLE_PATH + "/docs-samples-word-doc-count.pkl"
DOCS_SAMPLES_KPS_COUNT = SAMPLE_PATH + "/docs-samples-kps-count.pkl"
DOCS_SAMPLES_KPS_DOC_COUNT = SAMPLE_PATH + "/docs-samples-kps-doc-count.pkl"
# Extra - dictionaries by tokens or keyphrases, corpus in integer format
DOC_DIRECTORY = SAMPLE_PATH + "/doc_directory.pkl"
DOC_INVERSE_DIRECTORY = SAMPLE_PATH + "/doc_inverse_directory.pkl"
UNIQUE_TOKENS_COUNTS = SAMPLE_PATH + "/unique_tokens_counts.pkl"
UNIQUE_KEYPHRASES_COUNTS = SAMPLE_PATH + "/unique_keyphrases_counts.pkl"
DICTIONARY_TOKENS = SAMPLE_PATH + "/dictionary_tokens.pkl"
DICTIONARY_KEYPHRASES = SAMPLE_PATH + "/dictionary_keyphrases.pkl"
CORPUS_TOKENS = SAMPLE_PATH + "/corpus_tokens.pkl"
CORPUS_BAG_OF_WORDS = SAMPLE_PATH + "/corpus_bag_of_words.pkl"
CORPUS_KEYPHRASES = SAMPLE_PATH + "/corpus_keyphrases.pkl"
CORPUS_BAG_OF_KEYPHRASES = SAMPLE_PATH + "/corpus_bag_of_keyphrases.pkl"

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

# TF
DOCS_SAMPLES_TF = SAMPLE_PATH + "/docs-samples-tf.pkl"
DOCS_SAMPLES_TF_KPS = SAMPLE_PATH + "/docs-samples-tf-kps.pkl"

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

# TF
DOC_SAMPLE_A_TF = SAMPLE_PATH + "/doc-sample-a-tf.pkl"
DOC_SAMPLE_A_TF_KPS = SAMPLE_PATH + "/doc-sample-a-tf-kps.pkl"
DOC_SAMPLE_B_TF = SAMPLE_PATH + "/doc-sample-b-tf.pkl"
DOC_SAMPLE_B_TF_KPS = SAMPLE_PATH + "/doc-sample-b-tf-kps.pkl"

# SIMS
SIM_TF_TOKENS = SAMPLE_PATH + "/sim-tf-tokens.pkl"
SIM_TF_KEYPHRASES = SAMPLE_PATH + "/sim-tf-keyphrases.pkl"

SIM_TF_IDF_TOKENS = SAMPLE_PATH + "/sim-tf-idf-tokens.pkl"
SIM_TF_IDF_KEYPHRASES = SAMPLE_PATH + "/sim-tf-idf-keyphrases.pkl"

SIM_LSI_TOKENS = SAMPLE_PATH + "/sim-lsi-tokens.pkl"
SIM_LSI_KEYPHRASES = SAMPLE_PATH + "/sim-lsi-keyphrases.pkl"

SIM_LDA_TOKENS = SAMPLE_PATH + "/sim-lda-tokens.pkl"
SIM_LDA_KEYPHRASES = SAMPLE_PATH + "/sim-lda-keyphrases.pkl"

# Wikipedia
WIKIPEDIA_RESOURCES = RESOURCES_PATH + '/wikipedia'
PAGE_DICTIONARY = WIKIPEDIA_RESOURCES + '/page_dictionary.pkl'
PAGE_INVERSE_DICT = WIKIPEDIA_RESOURCES + '/page_inverse_dict.pkl'
PAGE_REDIRECTS = WIKIPEDIA_RESOURCES + '/page_redirects.pkl'
REDIRECT_PAGES = WIKIPEDIA_RESOURCES + '/redirect_pages.pkl'

HASHES_INTERSECTION = WIKIPEDIA_RESOURCES + '/hashes-intersection.pkl'

KEYPHRASEVARIATIONS_DOCS = WIKIPEDIA_RESOURCES + '/keyphrasevariations_docs.pkl'

def get_relative_path(data_path, extra_path):
    """Merge path"""
    return data_path + extra_path
