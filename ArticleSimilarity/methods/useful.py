""" Useful methods """
import tensorflow as tf

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
        print("Matrix reconstruction using %d singular value(s)" % (nsv),
              matrix_, matrix_singular_values)
        yield matrix_, matrix_singular_values

def wordvectors_centroid(wordvectors, labels, default_shape=True):
    """ Receive a gensim wordvector and a list of labels
    and returns a tensor of shape=(n,) with the mean of the labels's vectors"""
    vector_shape = (300,) if default_shape \
            else (wordvectors.get_vector(next(iter(wordvectors.vocab)))).shape
    mean = tf.zeros(vector_shape, dtype=tf.float64)
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

def tensor_to_value(tensor):
    """Receive tensor and return values"""
    with tf.Session() as sess:
        return sess.run(tensor)