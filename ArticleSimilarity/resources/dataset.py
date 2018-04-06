"""Methods to load resources"""
import sys
import os.path
import hashlib
import random
import pickle
# Whoosh
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, ID, KEYWORD, STORED
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter
from whoosh import qparser
from whoosh.qparser import QueryParser
# from whoosh.writing import BufferedWriter

INDEX_DATA = "/index-data"
SAMPLE_IDS = "/sample-ids"

# Default extension for aminer files in txt
EXT_AMINER_TXT = ".txt"

def get_default_analizer():
    """Return default analizer"""
    # Tokenizer from Whoosh
    analizer = RegexTokenizer() | LowercaseFilter() | StopFilter()
    return analizer

def get_default_sample_init():
    """Return deault values"""
    nrand = 100
    doc_limit = 10
    fixed_seed = 0
    return nrand, doc_limit, fixed_seed

def get_default_suffix(extra_suffix=""):
    """Return default suffix"""
    nrand, doc_limit, fixed_seed = get_default_sample_init()
    suffix_path = "-" + str(nrand) + "-" + str(doc_limit) + "-" + str(fixed_seed) + extra_suffix
    return suffix_path

def get_default_sample_path(data_path, sample_suffix=""):
    """Return path to sample with default parameters as suffix"""
    suffix_path = get_default_suffix(extra_suffix=sample_suffix) + ".bin"
    index_data_sample_path = data_path + SAMPLE_IDS + suffix_path
    return index_data_sample_path

def get_default_sample_path_related(data_path):
    """Return path to sample with default parameters as suffix"""
    return get_default_sample_path(data_path, sample_suffix="-related-fullcontent")

def get_default_sample_path_random(data_path):
    """Return path to sample with default parameters as suffix"""
    return get_default_sample_path(data_path, sample_suffix="-random-fullcontent")

def get_filenames(path_resources, ext=EXT_AMINER_TXT):
    """Receive a path to resources and yield a list of paths to files with extension 'ext'"""
    # Walk over files in the given path
    for (path_name, _, filenames) in os.walk(path_resources):
        for filename in filenames:
            # Get extension of length len_ext
            extension = filename[-len(ext):]
            # If extension is  ext then yield the file's path
            if extension == ext:
                path_filename = os.path.join(path_name, filename)
                yield path_filename

def get_aminer_txt(pfilename):
    """Receive path to a file in aminer txt format and
    yield the fields of each document in the file."""
    # Open file for reading
    fin = open(pfilename, "r", encoding="utf-8")
    # Initialization of variables
    indexdoc, indexdoc_tmp, title, text = u"", u"", u"", u""
    # Iterate over lines of documents in aminer format
    for line in fin:
        # Remove new lines
        ldoc = line.strip("\n")
        # Check if the line is not empty
        if ldoc:
            # Read target document's fields
            if ldoc[:2] == "#*": # Title
                title = ldoc[2:]
            if ldoc[:6] == "#index": # Index
                indexdoc_tmp = ldoc[1:]
            if ldoc[:2] == "#!": # Abstract
                text = ldoc[2:]
        elif indexdoc_tmp and indexdoc != indexdoc_tmp:
            indexdoc = indexdoc_tmp
            # Yield document's fields
            # indexdoc, title, title + content
            yield indexdoc, title, title if not text else title + " " + text
            title, text = u"", u""

def index_aminer_txt_in(documents_path):
    """Receive path to aminer resource and index each document with Whoosh"""
    # Batch of data to index
    data_batch = 50000
    # Analizer
    analizer = get_default_analizer()
    # Path to indexed data
    whoosh_index = documents_path + INDEX_DATA
    # Create path if it doesn't exists
    if not os.path.exists(whoosh_index):
        os.mkdir(whoosh_index)
        # Define schema for indexed data
        schema = Schema(indexdoc=ID(stored=True, unique=True), # indexdoc
                        title=STORED(), # title
                        content=STORED(), # content
                        bag_of_words=KEYWORD(scorable=True), # BoW - Not stored
                        bag_of_words_hash=ID(stored=True), # content hash
                        bag_of_words_title=KEYWORD(scorable=True), # BoW title - Not stored
                        cardinality=STORED()) # Cardinality
        # Create index
        ix_aminer = create_in(whoosh_index, schema)
        # Define writer
        writer = ix_aminer.writer(limitmb=2048)
        # For each file in aminer path
        for pfilename in get_filenames(documents_path):
            # Counter
            i = 1
            # For each document in the file
            for indexdoc, title, text in get_aminer_txt(pfilename):
                # Get string with a subset of words in lowercase
                # from the title and the content of the document
                tokens = " ".join(set([t.text for t in analizer(text)]))
                # Get string with a subset of words from title
                tokens_title = " ".join(set([t.text for t in analizer(title)]))
                # Get number of words
                cardinality = tokens.count(" ") + 1
                # Write document to the index
                writer.add_document(indexdoc=indexdoc,
                                    title=title,
                                    content=text,
                                    bag_of_words=tokens,
                                    bag_of_words_hash=hashlib.md5(tokens.encode()).hexdigest(),
                                    bag_of_words_title=tokens_title,
                                    cardinality=cardinality)
                # Show number of documents each batch
                if i % data_batch == 0:
                    print("Documents: ", i, file=sys.stderr)
                i += 1
        # Commit data
        writer.commit()
    else:
        print("This index already exists: %s. \
                \n - Move or delete the path if you want to index again." % \
                (whoosh_index), file=sys.stderr)

def save_sample_aminer_related(data_path):
    """Receive path to resource and save a list of related documents"""
    # Analizer
    analizer = get_default_analizer()
    # Init variables and paths
    nrand, doc_limit, fixed_seed = get_default_sample_init()
    index_data_path = data_path + INDEX_DATA
    index_data_sample_path = get_default_sample_path_related(data_path)
    # If index exists
    if os.path.exists(index_data_path) and not os.path.exists(index_data_sample_path):
        print("Opening indexed data ...", file=sys.stderr)
        # Open index
        ix_data = open_dir(index_data_path)
        # Parser for queries grouping query terms with OR
        parser = QueryParser("bag_of_words", ix_data.schema, group=qparser.OrGroup)
        # Create reader for all documents
        with ix_data.reader() as reader:
            with ix_data.searcher() as searcher:
                print("Sampling related documents ...", file=sys.stderr)
                # Threshold to get nrand random documents
                # roulette_threshold = (2.0*nrand)/reader.doc_count()
                roulette_threshold = 0.5
                # Init pseudo-random generator
                random.seed(fixed_seed)
                print(" - Seed: %d, \
                        \n - Threshold: %f, \
                        \n - Random docs: %d, \
                        \n - Max related docs per random doc: %d, \
                        \n - Docs in dataset: %d" % (fixed_seed,
                                                     roulette_threshold,
                                                     nrand,
                                                     doc_limit,
                                                     reader.doc_count()), file=sys.stderr)
                # Content hash
                docs_hashes = {}
                # Iter documents
                for docid in reader.all_doc_ids():
                    # Minimum expected number of docs
                    if len(docs_hashes) >= nrand*doc_limit:
                        break
                    # Get stored fields from document
                    doc = reader.stored_fields(docid)
                    # Check if doc has content
                    if not doc_has_content(doc):
                        continue
                    # If random number is greater than threshold then document is not selected
                    if random.random() > roulette_threshold:
                        continue
                    # Get string with a subset of words
                    doc_title = " ".join([w.text for w in analizer(doc['title'])])
                    # Parse query with the subset of words in the title
                    query = parser.parse(doc_title)
                    # Query documents related with the title
                    result = searcher.search(query, limit=doc_limit)
                    for res in result:
                        if not doc_has_content(res):
                            continue
                        docs_hashes[res['bag_of_words_hash']] = res['indexdoc']
                # Dump document ids from sample
                with open(index_data_sample_path, "wb") as fout:
                    pickle.dump(list(docs_hashes.values()), fout)
                    print("Sample size: ", len(docs_hashes), file=sys.stderr)
    else:
        # Nothing is done.
        # If you want to resample, first check if index exists or remove sample file
        print("Sample of related documents already exists.\
                \n - Index path: %s \n - Sample path: %s" % (index_data_path,
                                                             index_data_sample_path),
              file=sys.stderr)

def save_sample_aminer_random(data_path):
    """Receive path to resource and save a list of random documents"""
    # Init variables and paths
    nrand, doc_limit, fixed_seed = get_default_sample_init()
    index_data_path = data_path + INDEX_DATA
    index_data_sample_path = get_default_sample_path_random(data_path)
    # If index exists
    if os.path.exists(index_data_path) and not os.path.exists(index_data_sample_path):
        print("Opening indexed data ...", file=sys.stderr)
        # Open index
        ix_data = open_dir(index_data_path)
        # Parser for queries grouping query terms with OR
        # parser = QueryParser("bag_of_words", ix_data.schema, group=qparser.OrGroup)
        # Create reader for all documents
        with ix_data.reader() as reader:
            print("Sampling random documents ...", file=sys.stderr)
            # Threshold to get nrand*doc_limit random documents
            expected_docs = nrand*doc_limit
            # roulette_threshold = 5.0*expected_docs/reader.doc_count()
            roulette_threshold = 0.5
            # Init pseudo-random generator
            random.seed(fixed_seed)
            print(" - Seed: %d, \
                    \n - Threshold: %f, \
                    \n - Random docs: %s, \
                    \n - Docs in dataset: %s" % (fixed_seed,
                                                 roulette_threshold,
                                                 expected_docs,
                                                 reader.doc_count()), file=sys.stderr)
            # Content by hash to avoid content repetition
            docs_hashes = {}
            # Iter documents
            for docid in reader.all_doc_ids():
                # Stop condition
                if len(docs_hashes) >= expected_docs:
                    break
                # Get stored fields from document
                doc = reader.stored_fields(docid)
                if not doc_has_content(doc):
                    continue
                # If random number is greater than threshold then document is not selected
                if random.random() > roulette_threshold:
                    continue
                docs_hashes[doc['bag_of_words_hash']] = doc['indexdoc']
            # Dump document ids from sample
            with open(index_data_sample_path, "wb") as fout:
                pickle.dump(list(docs_hashes.values()), fout)
                print("Random sample size: ", len(docs_hashes), file=sys.stderr)
    else:
        # Nothing is done.
        # If you want to resample, first check if index exists or remove sample file
        print("Sample of random documents already exists.\
                    \n - Index path: %s \
                    \n - Sample path: %s" % (index_data_path, index_data_sample_path),
              file=sys.stderr)

def doc_has_content(doc):
    """Check if doc has content"""
    return len(doc['title']) < len(doc['content'])

def get_sample_ids(data_path, related_docs=True):
    """Return doc_ids for sample"""
    if related_docs:
        docs_ids = get_docidssample_aminer_related(data_path)
    else:
        docs_ids = get_docidssample_aminer_random(data_path)
    return docs_ids

def get_docidssample_aminer(index_data_sample_path):
    """Return docs ids from sample of documents"""
    # Open file if exists
    with open(index_data_sample_path, "rb") as fin:
        docs_ids = pickle.load(fin)
        print("Sample size: %d documents" % len(docs_ids), file=sys.stderr)
        print(" - Content from %s" % (index_data_sample_path), file=sys.stderr)
        return docs_ids[:10]

def get_docidssample_aminer_related(data_path):
    """Return docs ids from sample of related documents"""
    # Get sample default path
    index_data_sample_path = get_default_sample_path_related(data_path)
    return get_docidssample_aminer(index_data_sample_path)

def get_docidssample_aminer_random(data_path):
    """Return docs ids from sample of random documents"""
    # Get sample default path
    index_data_sample_path = get_default_sample_path_random(data_path)
    return get_docidssample_aminer(index_data_sample_path)

def get_sample_aminer(index_data_path, docs_ids):
    """Receive path to sample and return a list with docs ids and
    dictionary with documents from the sample"""
    # Analizer
    analizer = get_default_analizer()
    # If index exists
    if os.path.exists(index_data_path):
        # Open index
        ix_data = open_dir(index_data_path)
        # Field to query
        parser = QueryParser("indexdoc", ix_data.schema)
        # Search for each doc id
        with ix_data.searcher() as searcher:
            # Load documents to memory
            docs = {}
            for docid in docs_ids:
                # Query doc id
                query = parser.parse(docid)
                # Find and get the first result
                doc = searcher.search(query)[0]
                # Get bag of words from content
                bag_of_words = set([t.text for t in analizer(doc['content'])])
                # Save doc values in dict
                docs.setdefault(docid, {'bag_of_words': set(), 'cardinality': 0})
                docs[docid]['bag_of_words'] = bag_of_words
                docs[docid]['cardinality'] = doc['cardinality']
            return {'docs_ids': docs_ids, 'docs': docs}
    else:
        print("0 documents loaded \
                \n - Index doesn't exists", file=sys.stderr)
        return None

def get_sample_aminer_related(data_path):
    """Receive path to a resource and return a list with docs ids and
    dictionary with documents from the sample"""
    # Get index default path
    index_data_path = data_path + INDEX_DATA
    # Get docs ids
    docs_ids = get_docidssample_aminer_related(data_path)
    return get_sample_aminer(index_data_path, docs_ids)

def get_sample_aminer_random(data_path):
    """Receive path to a resource and return a list with docs ids and
    dictionary with documents from the sample"""
    # Get index default path
    index_data_path = data_path + INDEX_DATA
    # Get docs ids
    docs_ids = get_docidssample_aminer_random(data_path)
    return get_sample_aminer(index_data_path, docs_ids)
