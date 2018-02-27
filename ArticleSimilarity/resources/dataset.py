import sys, os.path
import hashlib
# Whoosh 
from whoosh.index import create_in
from whoosh.fields import Schema, ID, KEYWORD, STORED
from whoosh.analysis import StemmingAnalyzer, RegexTokenizer, LowercaseFilter, StopFilter
# from whoosh.writing import BufferedWriter

# Import default paths to indexed data
from .indexing import INDEX_DATA

# Default extension for aminer files in txt
EXT_AMINER_TXT = ".txt"
# Tokenizer from Whoosh 
analizer = RegexTokenizer() | LowercaseFilter() | StopFilter()

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
    for lf in fin:
        # Remove new lines 
        ldoc = lf.strip("\n")
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
        ix = create_in(whoosh_index, schema)
        # Define writer 
        writer = ix.writer(limitmb=2048)
        # For each file in aminer path
        for pfilename in get_filenames(documents_path):
            # Counter
            i = 1 
            # For each document in the file 
            for indexdoc, title, text in get_aminer_txt(pfilename):
                # Get string with set of words in lowercase from the title and the content of the document
                tokens = " ".join(set([t.text for t in analizer(text)]))
                # Get string with set of words from title 
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
                \nMove or delete the path if you want to index again." % (whoosh_index), file=sys.stderr)
