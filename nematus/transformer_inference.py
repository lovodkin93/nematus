"""Adapted from Nematode: https://github.com/demelin/nematode """

import numpy as np
import tensorflow as tf
# from gi.overrides.GObject import signal_accumulator_first_wins
# from parsing.corpus import ConllSent

# from transformer_layers import get_shape_list
# from transformer_layers import get_positional_signal
# from transformer_layers import get_tensor_from_times
# import util


# def sample(session, model, x, x_mask, graph=None):
#     """Randomly samples from a Transformer translation model.

#     Args:
#         session: TensorFlow session.
#         model: a Transformer object.
#         x: Numpy array with shape (factors, max_seq_len, batch_size).
#         x_mask: Numpy array with shape (max_seq_len, batch_size).
#         graph: a SampleGraph (to allow reuse if sampling repeatedly).

#     Returns:
#         A list of NumPy arrays (one for each input sentence in x).
# =======
try:
    from . import tf_utils
    from .tf_utils import get_shape_list
    from .transformer import INT_DTYPE, FLOAT_DTYPE
    from .transformer_layers import get_positional_signal, get_right_context_mask
    from .data_iterator import convert_text_to_graph
    from . import util
except (ModuleNotFoundError, ImportError) as e:
    import tf_utils
    import util
    from tf_utils import get_shape_list
    from transformer import INT_DTYPE, FLOAT_DTYPE
    from transformer_layers import get_positional_signal, get_right_context_mask
    from data_iterator import convert_text_to_graph


class EncoderOutput:

    def __init__(self, enc_output, cross_attn_mask):
        self.enc_output = enc_output
        self.cross_attn_mask = cross_attn_mask


class ModelAdapter:
    """Implements model-specific functionality needed by the *Sampler classes.

    The BeamSearchSampler and RandomSampler classes need to work with RNN and
    Transformer models, which have different interfaces (and obviously
    different architectures). This class hides the Transformer-specific details
    behind a common interace (see rnn_inference.ModelAdapter for the RNN
    counterpart).
    """

    def __init__(self, model, config, scope):
        self._model = model
        self._config = config
        self._scope = scope

    @property
    def model(self):
        return self._model

    @property
    def config(self):
        return self._config

    @property
    def target_vocab_size(self):
        return self._model.dec.embedding_layer.get_vocab_size()

    @property
    def batch_size(self):
        return tf.shape(self._model.inputs.x)[-1]

    def encode(self):
        with tf.name_scope(self._scope):
            enc_output, cross_attn_mask = self._model.enc.encode(
                self._model.source_ids, self._model.source_mask)
            return EncoderOutput(enc_output, cross_attn_mask)

    def generate_decoding_function(self, encoder_output):

        with tf.name_scope(self._scope):
            # Generate a positional signal for the longest possible output.
            positional_signal = get_positional_signal(
                self._config.translation_maxlen,
                self._config.embedding_size,
                FLOAT_DTYPE)

        decoder = self._model.dec

        def _decoding_function(step_target_ids, current_time_step, memories, x):
            """Single-step decoding function.

            Args:
                step_target_ids: Tensor with shape (batch_size)
                current_time_step: scalar Tensor.
                memories: dictionary (see top-level class description)

            Returns:
            """
            with tf.name_scope(self._scope):
                # TODO make sure the build is like in the train
                # TODO make sure it learns to end sentences at some point

                # print("memory print", memories)
                # TODO no memories in target_graph inference. and need to word
                # by word translate in train (perhaps with the same func?)
                printops = []
                printops.append(
                    tf.Print([], [tf.shape(x), x], "decoded x_", 10, 50))
                printops.append(
                    tf.Print([], [current_time_step], "decoded current_time_step", 10, 50))
                printops.append(
                    tf.Print([], [tf.shape(step_target_ids), step_target_ids], "step_target_ids", 10, 50))
                printops.append(tf.Print([], [memories["layer_1"]["keys"], memories[
                                "layer_1"]["values"]], "memories", 10, 50))
                with tf.control_dependencies(printops):
                    if self.config.target_graph:
                        max_size = self.config.maxlen + 1
                        x = tf.pad(x, [[0, 0], [0, max_size - tf.shape(x)[1]]])
                        target_embeddings = decoder._embed(x)
                        signal_slice = positional_signal[
                            :, :current_time_step, :]
                        emb_shape = tf.shape(target_embeddings)
                        signal_slice = tf.pad(
                            signal_slice, [[0, 0], [0, emb_shape[1] - current_time_step], [0, 0]])
                    else:
                        target_embeddings = decoder._embed(step_target_ids)
                        signal_slice = positional_signal[
                            :, current_time_step - 1:current_time_step, :]

                # Add positional signal.
                printops = []
                printops.append(
                    tf.Print([], [tf.shape(signal_slice), signal_slice], "signal_slice", 10, 50))
                printops.append(tf.Print([], [tf.shape(
                    target_embeddings), target_embeddings], "target_embeddings", 10, 50))
                with tf.control_dependencies(printops):
                    target_embeddings += signal_slice
                # Optionally, apply dropout to embeddings.
                if self.config.transformer_dropout_embeddings > 0:
                    target_embeddings = tf.layers.dropout(
                        target_embeddings,
                        rate=self.config.transformer_dropout_embeddings,
                        training=decoder.training)

                layer_output = target_embeddings

                # add target graph created so far
                if self.config.target_graph:
                    for layer_id in range(self.config.target_gcn_layers):
                        # this or perhaps the more efficient tf based
                        # .compat.v1.py_function?
                        edges, labels = tf.compat.v1.py_func(
                            self.extract_graph, [x], [tf.float32, tf.float32], stateful=False)
                        edges.set_shape([None, max_size, max_size, 3])
                        labels.set_shape(
                            [None, max_size, max_size, self.config.target_labels_num])
                        # tf.ensure_shape(edges, [None, max_size, max_size, 3])
                        # tf.ensure_shape(
                        #     labels, [None, max_size, max_size, self.config.target_labels_num])
                        edges = util.dense_to_sparse_tensor(edges)
                        labels = util.dense_to_sparse_tensor(labels)
                        printops = []
                        printops.append(tf.Print([], [tf.shape(edges), edges.indices, tf.ones_like(
                            edges.values)], "pythoned edges", 10, 50))
                        printops.append(tf.Print([], [tf.shape(labels), labels.indices, tf.ones_like(
                            labels.values)], "pythoned labels", 10, 50))
                        printops.append(tf.Print([], [tf.shape(layer_output), layer_output], "last layer output" + str(
                            layer_id - 1) + " (is padded tp btch,120,emb?", 10, 50))
                        with tf.control_dependencies(printops):
                            inputs = [layer_output, edges, labels]
                            layer_output = self.model.dec.gcn_stack[
                                layer_id].apply(inputs)
                            layer_output += inputs[0]  # residual connection
                    # slice tensor to save space
                    printops = []
                    printops.append(tf.Print([], [tf.shape(layer_output), layer_output], "slicing gcn output", 10, 50))
                    with tf.control_dependencies(printops):
                        layer_output = layer_output[:, :current_time_step, :]
                    # Propagate values through the decoder stack.
                # NOTE: No self-attention mask is applied at decoding, as
                #       future information is unavailable.
                for layer_id in range(1, self.config.transformer_dec_depth + 1):
                    layer = decoder.decoder_stack[layer_id]
                    mem_key = 'layer_{:d}'.format(layer_id)
                    if self.config.target_graph:
                        layer_memories = memories[mem_key]
                    else:
                        layer_memories = None
                    self_attn_mask = None
                    printops = []
                    printops.append(tf.Print([], [tf.shape(layer_output), layer_output], "infer self_attending"+ str(layer_id), 10, 50))
                    with tf.control_dependencies(printops):
                        layer_output, memories[mem_key] = \
                            layer['self_attn'].forward(
                                layer_output, None, self_attn_mask, layer_memories)
                    printops = []
                    printops.append(tf.Print([], [tf.shape(layer_output), layer_output], "infer cross_attending" + str(layer_id), 10, 50))
                    with tf.control_dependencies(printops):
                        layer_output, _ = layer['cross_attn'].forward(
                            layer_output, encoder_output.enc_output,
                            encoder_output.cross_attn_mask)
                    layer_output = layer['ffn'].forward(layer_output)
                # Return prediction at the final time-step to be consistent
                # with the inference pipeline.
                printops = []
                printops.append(
                    tf.Print([], [tf.shape(layer_output), layer_output[:, -1, :]],
                             "is :,-1,:, in layer_output the logits of the last step (or rather of eos?)", 10, 50))

                with tf.control_dependencies(printops):
                    # keep only the logits of the newly predicted word
                    dec_output = layer_output[:, -1, :]
                # Project decoder stack outputs and apply the soft-max
                # non-linearity.
                step_logits = \
                    decoder.softmax_projection_layer.project(dec_output)
                return step_logits, memories

        return _decoding_function

    def generate_initial_memories(self, batch_size, beam_size):
        with tf.name_scope(self._scope):
            state_size = self.config.state_size
            memories = {}
            for layer_id in range(1, self.config.transformer_dec_depth + 1):
                memories['layer_{:d}'.format(layer_id)] = {
                    'keys': tf.tile(tf.zeros([batch_size, 0, state_size]),
                                    [beam_size, 1, 1]),
                    'values': tf.tile(tf.zeros([batch_size, 0, state_size]),
                                      [beam_size, 1, 1])
                }
            return memories

    def get_memory_invariants(self, memories):
        """Generate shape invariants for memories.

        Args:
            memories: dictionary (see top-level class description)

        Returns:
            Dictionary of shape invariants with same structure as memories.
        """
        with tf.name_scope(self._scope):
            invariants = dict()
            for layer_id in memories.keys():
                layer_mems = memories[layer_id]
                invariants[layer_id] = {
                    key: tf.TensorShape(
                        [None] * len(tf_utils.get_shape_list(layer_mems[key])))
                    for key in layer_mems.keys()
                }
            return invariants

    def gather_memories(self, memories, gather_coordinates):
        """ Gathers layer-wise memory tensors for selected beam entries.

        Args:
            memories: dictionary (see top-level class description)
            gather_coordinates: Tensor with shape [batch_size_x, beam_size, 2]

        Returns:
            Dictionary containing gathered memories.
        """
        with tf.name_scope(self._scope):

            shapes = {gather_coordinates: ('batch_size_x', 'beam_size', 2)}
            tf_utils.assert_shapes(shapes)

            coords_shape = tf.shape(gather_coordinates)
            batch_size_x, beam_size = coords_shape[0], coords_shape[1]

            def gather_attn(attn):
                # TODO Specify second and third?
                shapes = {attn: ('batch_size', None, None)}
                tf_utils.assert_shapes(shapes)
                attn_dims = tf_utils.get_shape_list(attn)
                new_shape = [beam_size, batch_size_x] + attn_dims[1:]
                tmp = tf.reshape(attn, new_shape)
                flat_tensor = tf.transpose(tmp, [1, 0, 2, 3])
                tmp = tf.gather_nd(flat_tensor, gather_coordinates)
                tmp = tf.transpose(tmp, [1, 0, 2, 3])
                gathered_values = tf.reshape(tmp, attn_dims)
                return gathered_values

            gathered_memories = dict()

            for layer_key in memories.keys():
                layer_dict = memories[layer_key]
                gathered_memories[layer_key] = dict()

                for attn_key in layer_dict.keys():
                    attn_tensor = layer_dict[attn_key]
                    gathered_memories[layer_key][attn_key] = \
                        gather_attn(attn_tensor)

            return gathered_memories

    def extract_graph(self, ids):
        # if len(ids.shape) == 1:
        #     ids = [ids]
        label_dict = self.model.target_labels_dict
        inv_dict = {v: k for k, v in label_dict.items()}
        print("ids", ids)
        # print("ids_shape", ids.shape)
        sents = [[inv_dict[idn] for idn in row if inv_dict[idn] not in ["<EOS>"]] # are allowed as words "<GO>", "<UNK>"
                 for row in ids]  # TODO is this indeed the right format of all the inputs?
        # strs = [inv_dict[idn] for idn in ids if inv_dict[idn] not in
        # ["<EOS>", "<GO>", "<UNK>"]] #TODO is this indeed the right format of
        # all the inputs?
        print("sents for graph", sents)
        converted = [convert_text_to_graph(
            sent, self._config.maxlen + 1, label_dict, self.model.target_labels_num, graceful=True) for sent in sents]
        edge_times, label_times = zip(*converted)
        # if len(edge_times.shape) == 3:
        #     edge_times = np.expand_dims(edge_times, axis=0)
        # if len(label_times.shape) == 3:
        #     label_times = np.expand_dims(label_times, axis=0)
        return edge_times, label_times
