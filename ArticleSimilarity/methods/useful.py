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
