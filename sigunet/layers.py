#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2DTranspose, Reshape, Concatenate

class Conv1DTranspose(Layer):

    def __init__(self, filters, kernel_size, stride=1):
        super(Conv1DTranspose, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.layer = Conv2DTranspose(filters=filters,
                                     kernel_size=(kernel_size, 1),
                                     strides=(stride, 1),
                                     padding='same',
                                     activation=tf.nn.relu)

    def call(self, x):

        output = Reshape((int(x.get_shape()[1]), 1, int(x.get_shape()[2])))(x)
        output = self.layer(output)
        output = Reshape((self.stride*int(x.get_shape()[1]), self.filters))(output)

        return output

class ThresholdDecision(Layer):

    def __init__(self, continue_positions, threshold):
        super(ThresholdDecision, self).__init__()

        self.continue_positions = continue_positions
        self.threshold = threshold

    def _inner_loop(self, seqs, i, cur):
        subtract = seqs[:, i+self.continue_positions] - seqs[:, i]
        subtract = tf.expand_dims(subtract, 1)
        return seqs, i + 1, tf.concat([cur, subtract], axis=1)

    def _loop_cond(self, seqs, i, cur):
        return tf.equal(i, seqs.get_shape().as_list()[1] - self.continue_positions - 1)

    def call(self, x):
        idx = tf.constant(0)
        cur = tf.zeros(tf.shape(x)[0])
        cur = tf.expand_dims(cur, 1)

        seqs = x[:, :, 2]

        seqs = tf.where(seqs > self.threshold, tf.ones(tf.shape(seqs)), tf.zeros(tf.shape(seqs)))
        seqs = tf.math.cumsum(seqs, axis=1)
        _, _, res = tf.while_loop(cond=self._loop_cond,
                                  body=self._inner_loop,
                                  loop_vars=[seqs, idx, cur],
                                  shape_invariants=[seqs.get_shape(), idx.get_shape(), tf.TensorShape([None, None])])
        res = tf.equal(res, self.continue_positions)
        res = tf.reduce_any(res, axis=1)
        res = tf.cast(res, tf.int32)

        return res

class Vote(Layer):

    def __init__(self, ratio=0.5):
        super(Vote, self).__init__()

        self.ratio = ratio

    def call(self, x):

        output = Concatenate(axis=-1)(x)
        output = tf.cast(output, tf.float32)
        output = tf.reduce_mean(output, axis=-1)
        output = tf.greater(output, self.ratio)

        return tf.cast(output, tf.int32)


if __name__ == '__main__':

    thr_layer = ThresholdDecision(continue_positions=2, threshold=0.5)
    x = tf.constant([
        [[0.3, 0.6, 0.1], [0, 0, 0.6], [0., 0, 0.6]],
        [[0.3, 0.6, 0.1], [0, 0, 0.1], [0., 0, 0.6]],
        [[0.3, 0.6, 0.1], [0, 0, 0.6], [0., 0, 0.1]],
    ])
    res1 = thr_layer.call(x)
    res2 = thr_layer.call(x)
    sess = tf.Session()
    print(sess.run(res1))
    print(sess.run(res1))
