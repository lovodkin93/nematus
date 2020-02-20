import tensorflow as tf


class ModelInputs(object):
    def __init__(self, config):
        # variable dimensions
        seq_len, batch_size, mrt_sampleN = None, None, None
        # if config.target_graph:
        #     seq_len = config.maxlen
        # mrt_sampleN = batch_size X sampleN
        self.x = tf.compat.v1.placeholder(
            name='x',
            shape=(config.factors, seq_len, batch_size),
            dtype=tf.int32)

        self.x_mask = tf.compat.v1.placeholder(
            name='x_mask',
            shape=(seq_len, batch_size),
            dtype=tf.float32)

        self.y = tf.compat.v1.placeholder(
            name='y',
            shape=(seq_len, batch_size),
            dtype=tf.int32)

        self.y_mask = tf.compat.v1.placeholder(
            name='y_mask',
            shape=(seq_len, batch_size),
            dtype=tf.float32)

        self.scores = tf.compat.v1.placeholder(
            name='scores',
            shape=(mrt_sampleN),
            dtype=tf.float32)

        self.index = tf.compat.v1.placeholder(
            name='index',
            shape=(mrt_sampleN),
            dtype=tf.int32)

        self.training = tf.compat.v1.placeholder_with_default(
            False,
            name='training',
            shape=())
        if config.target_graph:
            edge_labels_num = 3 # (self left right)
            if config.target_labels_num is None:
                raise ValueError("target_labels_num is not defined, please figure it by the dictionary and supply it as a flag")
            elif config.target_labels_num > 0:
                self.label_times = tf.compat.v1.sparse_placeholder(
                    name='labels',
                    shape=(seq_len, seq_len, config.target_labels_num, batch_size),
                    dtype=tf.float32)

            self.edge_times = tf.compat.v1.sparse_placeholder(
                name='edges',
                shape=(seq_len, seq_len, edge_labels_num, batch_size),
                dtype=tf.float32)
