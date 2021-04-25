"""Adapted from Nematode: https://github.com/demelin/nematode """

import numpy as np

import sys
import tensorflow as tf
import ast

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
                self._model.source_ids, self._model.source_mask, self._model.s_same_scene_mask, self._model.s_parent_scaled_mask, self._model.s_UD_distance_scaled_mask)
            ################################################ PRINTS #################################################
            # print_ops = []
            # print_ops.append(tf.compat.v1.Print([], [tf.shape(self._model.s_same_scene_mask), self._model.s_same_scene_mask], "AVIVSL30: s_same_scene_mask:", summarize=10000))
            # with tf.control_dependencies(print_ops):
            #     enc_output = enc_output * 1
            #########################################################################################################
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
                # TODO no memories in target_graph inference. and need to words
                # by words translate in train (perhaps with the same func?)
                printops = []
                # printops.append(
                #     tf.compat.v1.Print([], [tf.shape(x), x], "decoded x_", 10, 50))
                # printops.append(
                #     tf.compat.v1.Print([], [current_time_step], "decoded current_time_step", 10, 50))
                # printops.append(
                #     tf.compat.v1.Print([], [tf.shape(step_target_ids), step_target_ids], "step_target_ids", 10, 50))
                # printops.append(tf.compat.v1.Print([], [memories["layer_1"]["keys"], memories[
                #                 "layer_1"]["values"]], "memories", 10, 50))
                # with tf.control_dependencies(printops):
                if self.config.target_graph or not self.config.sequential:
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
                target_embeddings += signal_slice
                # Optionally, apply dropout to embeddings.
                if self.config.transformer_dropout_embeddings > 0:
                    target_embeddings = tf.compat.v1.layers.dropout(
                        target_embeddings,
                        rate=self.config.transformer_dropout_embeddings,
                        training=decoder.training)

                layer_output = target_embeddings

                # add target graph created so far
                if self.config.target_graph and (self.config.target_gcn_layers > 0 or self.config.parent_head):
                    with tf.compat.v1.name_scope('infer_graph'):
                        edges, labels, parents = tf.compat.v1.py_func(
                            self.extract_graph, [x], [tf.float32, tf.float32, tf.float32], stateful=False)
                        if self.config.target_gcn_layers > 0:
                            edges.set_shape([None, max_size, max_size, 3])
                            edges = util.dense_to_sparse_tensor(edges)
                            edges = tf.sparse.SparseTensor(edges.indices, tf.ones_like(edges.values),
                                                           edges.dense_shape)  # make zeros from times
                            if self.config.target_labels_num > 0:
                                labels.set_shape(
                                    [None, max_size, max_size, self.config.target_labels_num])
                                # labels= tf.transpose(labels, perm=[len(labels.shape) - 1] + list(range(len(labels.shape) - 1)))
                                labels = util.dense_to_sparse_tensor(labels)
                                labels = tf.sparse.SparseTensor(labels.indices, tf.ones_like(labels.values),
                                                                labels.dense_shape)  # make zeros from times
                                # print_ops = []
                                # print_ops.append(tf.compat.v1.Print([], [tf.shape(labels)], "labels shape", 100, 200))
                                # print_ops.append(tf.compat.v1.Print([], [tf.shape(edges)], "edges shape", 100, 200))
                                # with tf.control_dependencies(print_ops):
                                #     labels = labels * 1
                            for layer_id in range(self.config.target_gcn_layers):
                                if self.config.target_labels_num > 0:
                                    inputs = [layer_output, edges, labels]
                                else:
                                    inputs = [layer_output, edges]
                                layer_output = decoder.gcn_stack[
                                    layer_id].apply(inputs)
                                layer_output += inputs[0]  # residual connection

                if self.config.target_graph or not self.config.sequential:
                    layer_output = layer_output[:, :current_time_step, :]
                    # Propagate values through the decoder stack.
                # NOTE: No self-attention mask is applied at decoding, as
                #       future information is unavailable.
                with tf.compat.v1.name_scope('infer_atten_mask'):
                    self_attn_mask = None
                    attention_rules = []
                    if self.config.parent_head:
                        # make zeros from times
                        parents = tf.convert_to_tensor(parents)
                        parents.set_shape([None, max_size, max_size])
                        parents = parents[:, :current_time_step, :current_time_step]
                        parents = parents - self._config.translation_maxlen
                        parents = tf.maximum(parents, 0)
                        parents = self.model.process_parents(parents)
                        # add a function that inputs this (will be needed also in infer)
                        attention_rules.append(parents)
                    if attention_rules:
                        self_attn_mask = tf.zeros_like(attention_rules[0])  # allow attention
                        self_attn_mask = decoder.combine_attention_rules(attention_rules, self_attn_mask)

                s_same_scene_mask = self._model.s_same_scene_mask
                if s_same_scene_mask is not None:
                    s_same_scene_mask = tf.cast(s_same_scene_mask, dtype=tf.float32)

                if self.config.source_same_scene_cross_attention_head:
                    if self.config.source_same_scene_cross_attention_masks_layers == "all_layers":
                        source_same_scene_cross_attention_masks_layers = list(range(1, self.config.transformer_dec_depth + 1))
                    else:
                        source_same_scene_cross_attention_masks_layers = ast.literal_eval(
                            self.config.source_same_scene_cross_attention_masks_layers)
                else:
                    source_same_scene_cross_attention_masks_layers = []


                for layer_id in range(1, self.config.transformer_dec_depth + 1):
                    layer = decoder.decoder_stack[layer_id]
                    mem_key = 'layer_{:d}'.format(layer_id)
                    if self.config.target_graph or not self.config.sequential:
                        if self.config.target_graph and self.config.sequential:
                            raise NotImplementedError(
                                "need to add attention mask to prevent words from forward attention in target)graph inference")
                        layer_memories = None
                    else:
                        layer_memories = memories[mem_key]

                    cross_attention_keys_update_rules = []
                    if s_same_scene_mask is not None and layer_id in source_same_scene_cross_attention_masks_layers:
                        same_scene_cross_attention_key = tf.matmul(s_same_scene_mask, encoder_output.enc_output) / tf.cast(
                            tf.shape(encoder_output.enc_output)[1], dtype=tf.float32)
                        same_scene_cross_attention_key = tf.expand_dims(same_scene_cross_attention_key, axis=1)
                        cross_attention_keys_update_rules.append((same_scene_cross_attention_key, self.config.source_num_same_scene_cross_attention_head))
                    cross_attention_keys_update_rules = cross_attention_keys_update_rules if cross_attention_keys_update_rules else None  # if list is empty, make it None

                    # ############################################## PRINTING #######################################################
                    # printops = []
                    # if cross_attention_keys_update_rules is not None:
                    #     printops.append(tf.compat.v1.Print([], [layer_id],"AVIVSL9: cross_attention_keys_update_rules[0][0] ",summarize=10000))
                    # printops.append(tf.compat.v1.Print([], [cross_attention_keys_update_rules[0][1]], "AVIVSL9: cross_attention_keys_update_rules[0][0] ", summarize=10000))
                    # printops.append(tf.compat.v1.Print([], [tf.shape(cross_attention_keys_update_rules)[0][1]], "AVIVSL9: cross_attention_keys_update_rules[1] ", summarize=10000))
                    # printops.append(tf.compat.v1.Print([], [tf.shape(s_same_scene_mask)],"AVIVSL9: s_same_scene_mask ", summarize=10000))
                    # printops.append(tf.compat.v1.Print([], [tf.shape(encoder_output.enc_output)],
                    #                                    "AVIVSL9: encoder_output.enc_output ", summarize=10000))
                    # printops.append(tf.compat.v1.Print([], [tf.shape(same_scene_cross_attention_key)],
                    #                                    "AVIVSL9: same_scene_cross_attention_key after", summarize=10000))
                    # printops.append(tf.compat.v1.Print([], [tf.cast(tf.shape(encoder_output.enc_output)[1], dtype=tf.float32)],
                    #                                    "AVIVSL9: tf.cast(tf.shape(encoder_output.enc_output)[1], dtype=tf.float32) ", summarize=10000))
                    # with tf.control_dependencies(printops):
                    #     layer_output = layer_output * 1
                    # ###############################################################################################################

                    layer_output, memories[mem_key], _ = \
                        layer['self_attn'].forward(
                            layer_output, None, self_attn_mask, None, None, None, layer_memories, isDecoder=True) #AVIVSL: here send to the self-attention
                    layer_output, _, _ = layer['cross_attn'].forward(
                        layer_output, encoder_output.enc_output,
                        encoder_output.cross_attn_mask, None, None, cross_attention_keys_update_rules=cross_attention_keys_update_rules) # TODO: AVIVSL add here the cross_attention_keys_update_rules
                    layer_output = layer['ffn'].forward(layer_output)
                # Return prediction at the final time-step to be consistent
                # with the inference pipeline.
                # keep only the logits of the newly predicted words
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
                if self.config.target_graph:  # when target graph (conditional) no memories are kept
                    layer_memories = {}
                else:
                    layer_memories = {
                        'keys': tf.tile(tf.zeros([batch_size, 0, state_size]),
                                        [beam_size, 1, 1]),
                        'values': tf.tile(tf.zeros([batch_size, 0, state_size]),
                                          [beam_size, 1, 1])}
                memories['layer_{:d}'.format(layer_id)] = layer_memories
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
                if layer_dict is not None:  # when using self.config.target_graph memories are not used
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
            sents.append(extracted)  # are allowed as words "<GO>", "<UNK>"
            # TODO is this indeed the right format of all the inputs?
        # strs = [inv_dict[idn] for idn in ids if inv_dict[idn] not in
        # ["<EOS>", "<GO>", "<UNK>"]] #TODO is this indeed the right format of
        # all the inputs?
        # print("first sent for graph", sents[0])
        converted = [convert_text_to_graph(
            sent, self._config.maxlen + 1, self.model.target_labels_dict, self.model.target_labels_num, attend_max=True,
            graceful=True)
            for sent in sents]
        edge_times, label_times, parents = zip(*converted)
        edge_times = np.array(edge_times)
        # max_sen_len = max([len(parent) for parent in parents])  # sentences may have finished
        # if len(parents.shape) < 3:
        #     print(f"sents {sents}")
        # print(f"max_sen_len {max_sen_len}")
        # print(f"unmaxed last parents {parents[-3:]}")
        # same_sized_parents = []
        # for parent in parents:
        #     if parent.shape != 2:
        #         np.expand_dims(parent, 1)
        #     padding = max_sen_len - len(parent)
        #     parent = np.pad(parent, ((0, padding), (0, padding)), mode="constant", constant_values=float("inf"))
        #     same_sized_parents.append(parent)
        # # parents = [np.pad(parent, ((0, max_sen_len - len(parent)), (0, max_sen_len - len(parent))) ]
        # print(f"parents maxed {parents}")
        parents = np.array(parents, dtype=np.float32)
        # print("parents, edge shapes", parents.shape, edge_times.shape)
        # padding = edge_times.shape[1] - parents.shape[1]
        # print("pad size", padding)
        # parents = np.pad(parents, ((0, 0), (0, padding), (0, padding)), mode="constant")
        # print("returning", edge_times, label_times, parents)
        # print("returning types", type(edge_times), type(label_times), type(parents[0]))
        # print("returning parents len, supposed", len(parents), len(sents))
        # if len(edge_times.shape) == 3:
        #     edge_times = np.expand_dims(edge_times, axis=0)
        # if len(label_times.shape) == 3:
        #     label_times = np.expand_dims(label_times, axis=0)
        return edge_times, label_times, parents
