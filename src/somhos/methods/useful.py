""" Useful methods """
import sys
import time
import gc
import hashlib as hl
from pathlib import Path
import pickle
import tensorflow as tf
import numpy as np

def get_svd_reconstructions(sess, matrices_list, n_elements, nsv=1):
    """ Receive a list of matrices of same shape to reconstruct using nsv singular values. """
    variable_matrix = tf.Variable(tf.zeros((n_elements, n_elements), dtype=tf.float64),
                                  dtype=tf.float64, name='A')
    svd_sigma, svd_u, svd_v = tf.svd(variable_matrix) #SVD
    # UDV' matrix reconstruction with ns singular values
    variable_matrix_ = tf.matmul(svd_u[:, :nsv],
                                 tf.matmul(tf.diag(svd_sigma[:nsv]),
                                           svd_v[:, :nsv], adjoint_b=True))
    # M_ = tf.matmul(u[:,:ns], v[:,:ns], adjoint_b=True) # UV'
    sess.run(tf.global_variables_initializer())

    for matrix in matrices_list:
        tf_op = variable_matrix.assign(matrix)
        sess.run(tf_op)
        matrix_ = tf.convert_to_tensor(sess.run(variable_matrix_)) # Reconstruction of the matrix A
        matrix_singular_values = tf.convert_to_tensor(sess.run(svd_sigma)) # Singular values
        print_log(("Matrix reconstruction using %d singular value(s)" % (nsv),
                   matrix_, matrix_singular_values))
        yield matrix_, matrix_singular_values

def wordvectors_centroid(wordvectors, labels, default_shape=True):
    """ Receive a gensim wordvector and a list of labels
    and returns a tensor of shape=(n,) with the mean of the labels's vectors"""
    vector_shape = (300,) if default_shape \
            else (wordvectors.get_vector(next(iter(wordvectors.vocab)))).shape
    mean = np.zeros(vector_shape, dtype=np.float64)
    labels_iter = iter(labels)
    label_count = 0
    label = True
    while label != None:
        label = next(labels_iter, None)
        if label and label in wordvectors.vocab:
            mean += wordvectors.get_vector(label)
            label_count += 1
    if label_count > 0:
        mean /= label_count
    return mean

def n_similarity(labels, predictions):
    """ Receive two list of words and tensor with the
    cosine similarity between means of the words's vectors"""
    vector1 = tf.nn.l2_normalize(labels, axis=0)
    vector2 = tf.nn.l2_normalize(predictions, axis=0)
    cossim_v1v2 = 1 - tf.losses.cosine_distance(vector1, vector2, axis=0)
    return cossim_v1v2

def tensor_to_value(graph):
    """Receive tensor and return values"""
    values = None
    with tf.Session() as sess:
        values = sess.run(graph)
        sess.close()
    gc.collect()
    return values

def print_log(log, cond=True, echo=True, persistent=True, file=sys.stderr):
    """Save log"""
    if cond:
        if echo:
            print(log, file=file)
        if persistent:
            with open("article-similarity.log", "a") as flog:
                log = "\n" + time.strftime("%Y/%m/%d %H:%M ", time.gmtime()) + str(log)
                flog.write(log)

def lower_utf8(phrase):
    """Receive string and return utf-8 string lowered"""
    return phrase.lower().encode('utf-8')

def hash_16bytes(phrase):
    """Receive string and return first 16 digits of its md5"""
    phrase_hash_16 = hl.md5(phrase).hexdigest()[:16]
    return phrase_hash_16

def load_pickle(pickle_path):
    """Receive path and load pickle"""
    if Path(pickle_path).exists():
        with open(pickle_path, "rb") as fin:
            pickle_data = pickle.load(fin)
    else:
        print_log("Path do not exists: %s" % pickle_path, persistent=False)
        pickle_data = None
    return pickle_data

def save_pickle(data, pickle_path):
    """Receive path and save data to pickle file"""
    if not Path(pickle_path).exists():
        with open(pickle_path, "wb") as fout:
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
    else:
        print_log("Path exists: %s" % pickle_path, persistent=False)
