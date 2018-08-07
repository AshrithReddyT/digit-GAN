import tensorflow as tf

def generator(x,training=False,reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # 1st hidden layer
        conv1 = conv2d_transpose(x, 512, [4, 4], (1, 1), 'valid')
        conv1 = lrelu(tf.layers.batch_normalization(conv1, training=training), 0.2)

        # 2nd hidden layer
        conv2 = conv2d_transpose(conv1, 256, [4, 4], (2, 2), 'same')
        conv2 = lrelu(tf.layers.batch_normalization(conv2, training=training), 0.2)

        # 3rd hidden layer
        conv3 = conv2d_transpose(conv2, 128, [4, 4], (2, 2), 'same')
        conv3 = lrelu(tf.layers.batch_normalization(conv3, training=training), 0.2)

        # 4th hidden layer
        conv4 = conv2d_transpose(conv3, 64, [4, 4], (2, 2), 'same')
        conv4 = lrelu(tf.layers.batch_normalization(conv4, training=training), 0.2)

        # output layer
        conv5 = conv2d_transpose(conv4, 1, [4, 4], (2, 2), 'same')
        out = tf.nn.tanh(conv5)

        return out

def discriminator(x,training=False,reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = conv2d(x, 64, [4, 4], (2, 2), 'same')
        conv1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = conv2d(conv1, 128, [4, 4], (2, 2), 'same')
        conv2 = lrelu(tf.layers.batch_normalization(conv2, training=training), 0.2)

        # 3rd hidden layer
        conv3 = conv2d(conv2, 256, [4, 4], (2, 2), 'same')
        conv3 = lrelu(tf.layers.batch_normalization(conv3, training=training), 0.2)

        # 4th hidden layer
        conv4 = conv2d(conv3, 512, [4, 4], (2, 2), 'same')
        conv4 = lrelu(tf.layers.batch_normalization(conv4, training=training), 0.2)

        # output layer
        conv5 = conv2d(conv4, 1, [4, 4], (1, 1), 'valid')
        out = tf.nn.sigmoid(conv5)

        return out, conv5


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def conv2d(x, output_dim, kernel, stride, padding):
    return tf.layers.conv2d(x, output_dim, kernel, strides=stride, padding=padding)

def conv2d_transpose(x, output_dim, kernel, stride ,padding):
    return tf.layers.conv2d_transpose(x, output_dim, kernel, strides=stride, padding=padding)
