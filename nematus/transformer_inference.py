"""Adapted from Nematode: https://github.com/demelin/nematode """

import numpy as np

import sys
import tensorflow as tf
# from gi.overrides.GObject import signal_accumulator_first_wins
# from parsing.corpus import ConllSent

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

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
        return tf.shape(input=self._model.inputs.x)[-1]

    def encode(self):
        with tf.compat.v1.name_scope(self._scope):
            enc_output, cross_attn_mask = self._model.enc.encode(
                self._model.source_ids, self._model.source_mask)
            return EncoderOutput(enc_output, cross_attn_mask)

    def generate_decoding_function(self, encoder_output):

        with tf.compat.v1.name_scope(self._scope):
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
            with tf.compat.v1.name_scope(self._scope):
                # TODO make sure the build is like in the train
                # TODO make sure it learns to end sentences at some point

                # print("memory print", memories)
                # TODO no memories in target_graph inference. and need to word
                # by word translate in train (perhaps with the same func?)
                # printops = []
                # printops.append(
                #     tf.compat.v1.Print([], [tf.shape(x), x], "decoded x_", 10, 50))
                # printops.append(
                #     tf.compat.v1.Print([], [current_time_step], "decoded current_time_step", 10, 50))
                # printops.append(
                #     tf.compat.v1.Print([], [tf.shape(step_target_ids), step_target_ids], "step_target_ids", 10, 50))
                # printops.append(tf.compat.v1.Print([], [memories["layer_1"]["keys"], memories[
                #                 "layer_1"]["values"]], "memories", 10, 50))
                # with tf.control_dependencies(printops):
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
                    step_target_ids = tf.reshape(step_target_ids, [-1, 1])
                    target_embeddings = decoder._embed(step_target_ids)
                    signal_slice = positional_signal[
                        :, current_time_step - 1:current_time_step, :]

                # Add positional signal.
                # printops = []
                # printops.append(
                #     tf.compat.v1.Print([], [tf.shape(signal_slice), signal_slice], "signal_slice", 10, 50))
                # printops.append(tf.compat.v1.Print([], [tf.shape(
                #     target_embeddings), target_embeddings], "target_embeddings", 10, 50))
                # with tf.control_dependencies(printops):
                target_embeddings += signal_slice
                # Optionally, apply dropout to embeddings.
                if self.config.transformer_dropout_embeddings > 0:
                    target_embeddings = tf.compat.v1.layers.dropout(
                        target_embeddings,
                        rate=self.config.transformer_dropout_embeddings,
                        training=decoder.training)

                layer_output = target_embeddings

                # add target graph created so far
                if self.config.target_graph and self.config.target_gcn_layers > 0:
                    edges, labels = tf.compat.v1.py_func(
                        self.extract_graph, [x], [tf.float32, tf.float32], stateful=False)
                    edges.set_shape([None, max_size, max_size, 3])
                    edges = util.dense_to_sparse_tensor(edges)
                    if self.config.target_labels_num > 0:
                        labels.set_shape(
                            [None, max_size, max_size, self.config.target_labels_num])
                        labels = util.dense_to_sparse_tensor(labels)
                    for layer_id in range(self.config.target_gcn_layers):
                        if self.config.target_labels_num > 0:
                            inputs = [layer_output, edges, labels]
                        else:
                            inputs = [layer_output, edges]
                        layer_output = self.model.dec.gcn_stack[
                            layer_id].apply(inputs)
                        layer_output += inputs[0]  # residual connection

                if self.config.target_graph:
                    layer_output = layer_output[:, :current_time_step, :]
                    # Propagate values through the decoder stack.
                # NOTE: No self-attention mask is applied at decoding, as
                #       future information is unavailable.
                for layer_id in range(1, self.config.transformer_dec_depth + 1):
                    layer = decoder.decoder_stack[layer_id]
                    mem_key = 'layer_{:d}'.format(layer_id)
                    if self.config.target_graph:
                        layer_memories = None
                    else:
                        layer_memories = memories[mem_key]
                    self_attn_mask = None
                    layer_output, memories[mem_key] = \
                        layer['self_attn'].forward(
                            layer_output, None, self_attn_mask, layer_memories)
                    layer_output, _ = layer['cross_attn'].forward(
                        layer_output, encoder_output.enc_output,
                        encoder_output.cross_attn_mask)
                    layer_output = layer['ffn'].forward(layer_output)
                # Return prediction at the final time-step to be consistent
                # with the inference pipeline.
                # keep only the logits of the newly predicted word
                dec_output = layer_output[:, -1, :]
                # Project decoder stack outputs and apply the soft-max
                # non-linearity.
                step_logits = \
                    decoder.softmax_projection_layer.project(dec_output)
                return step_logits, memories

        return _decoding_function

    def generate_initial_memories(self, batch_size, beam_size):
        with tf.compat.v1.name_scope(self._scope):
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
        with tf.compat.v1.name_scope(self._scope):
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
        with tf.compat.v1.name_scope(self._scope):

            shapes = {gather_coordinates: ('batch_size_x', 'beam_size', 2)}
            tf_utils.assert_shapes(shapes)

            coords_shape = tf.shape(input=gather_coordinates)
            batch_size_x, beam_size = coords_shape[0], coords_shape[1]

            def gather_attn(attn):
                # TODO Specify second and third?
                shapes = {attn: ('batch_size', None, None)}
                tf_utils.assert_shapes(shapes)
                attn_dims = tf_utils.get_shape_list(attn)
                new_shape = [beam_size, batch_size_x] + attn_dims[1:]
                tmp = tf.reshape(attn, new_shape)
                flat_tensor = tf.transpose(a=tmp, perm=[1, 0, 2, 3])
                tmp = tf.gather_nd(flat_tensor, gather_coordinates)
                tmp = tf.transpose(a=tmp, perm=[1, 0, 2, 3])
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
        tokens_dict = self.model.target_tokens
        inv_dict = {v: k for k, v in tokens_dict.items()}
        # print("ids", ids[0,:])
        # print("ids_shape", ids.shape)
        sents = []
        for row in ids:
            extracted = []
            for idn in row:
                if idn not in inv_dict:
                    print("Label not understood, skipping", idn)
                elif inv_dict[idn] not in ["<EOS>"]:
                    extracted.append(inv_dict[idn])
            sents.append(extracted) # are allowed as words "<GO>", "<UNK>"
                   # TODO is this indeed the right format of all the inputs?
        # strs = [inv_dict[idn] for idn in ids if inv_dict[idn] not in
        # ["<EOS>", "<GO>", "<UNK>"]] #TODO is this indeed the right format of
        # all the inputs?
        # print("first sent for graph", sents[0])
        converted = [convert_text_to_graph(
            sent, self._config.maxlen + 1, self.model.target_labels_dict, self.model.target_labels_num, graceful=True) for sent in sents]
        edge_times, label_times = zip(*converted)
        # if len(edge_times.shape) == 3:
        #     edge_times = np.expand_dims(edge_times, axis=0)
        # if len(label_times.shape) == 3:
        #     label_times = np.expand_dims(label_times, axis=0)
        return edge_times, label_times
