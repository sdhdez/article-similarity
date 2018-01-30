import sys, os
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import numpy as np

# init
penn_tree_bank_tokenizer = TreebankWordTokenizer()
nltk_stopwords = set(stopwords.words('english'))
punctuation_table = str.maketrans("", "", string.punctuation)

def get_filenames(path_resources):
    file_count = 0
    for (path_name, _, filenames) in os.walk(path_resources):
        for filename in filenames:
            extension = filename[-4:]
            if extension == '.txt':
                file_count += 1
                path_filename = os.path.join(path_name, filename)
                yield path_filename

def get_target_text(pfilename):
    fin = open(pfilename, "r", encoding="utf-8")
    indexdoc, indexdoc_tmp, text = "", "", ""
    for lf in fin:
        ldoc = lf.strip("\n")
        if ldoc:
            if ldoc[:2] == "#*": # Title 
                text += ldoc[2:]
            if ldoc[:6] == "#index": # Index 
                indexdoc_tmp = ldoc[1:]
            if ldoc[:2] == "#!": # Abstract 
                text += ldoc[2:]
        elif indexdoc_tmp and indexdoc != indexdoc_tmp:
            indexdoc = indexdoc_tmp
            yield indexdoc, text
            text = ""

def filter_puntuation(text):
    return text.translate(punctuation_table)

def get_tokens_blanks(text):
    return [t for t in filter_puntuation(text).split(" ") if t]

def get_tokens_penn_tree_bank(text):
    return penn_tree_bank_tokenizer.tokenize(text)

def get_tokens_default(text):
    return set(get_tokens_blanks(text))

def filter_stopwords_nltk(tokens):
    return tokens - nltk_stopwords

def get_bag_of_words(tokens):
    return filter_stopwords_nltk(tokens)

def create_bow_from_files_in(documents_path, bag_of_words_file): 
    all_bag_of_words = []
    for pfilename in get_filenames(documents_path):
        if pfilename[-3:] != "txt":
            continue
        for indexdoc, text in get_target_text(pfilename):
            bag_of_words = get_bag_of_words(get_tokens_default(text.lower()))
            all_bag_of_words.append({ 'set': bag_of_words, 
                                'cardinality': len(bag_of_words),
                                'index': indexdoc
                                })
    return all_bag_of_words
