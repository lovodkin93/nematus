"""Adapted from Nematode: https://github.com/demelin/nematode """

import numpy as np
import tensorflow as tf
from parsing.corpus import ConllSent

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
    from .transformer_layers import get_positional_signal
    from .data_iterator import convert_text_to_graph
    from . import util
except (ModuleNotFoundError, ImportError) as e:
    import tf_utils
    import util
    from tf_utils import get_shape_list
    from transformer import INT_DTYPE, FLOAT_DTYPE
    from transformer_layers import get_positional_signal
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

#     def normalization_alpha(self):
#         return self._normalization_alpha


# def construct_sampling_ops(model):
#     """Builds a graph fragment for sampling over a TransformerModel.

#     Args:
#         model: a TransformerModel.

#     Returns:
#         A tuple (ids, scores), where ids is a Tensor with shape (batch_size,
#         max_seq_len) containing one sampled translation for each input sentence
#         in model.inputs.x and scores is a Tensor with shape (batch_size)
#     """
#     ids, scores = decode_greedy(model, do_sample=True)
#     return ids, scores


# def construct_beam_search_ops(models, beam_size, normalization_alpha):
#     """Builds a graph fragment for sampling over a TransformerModel.

#     Args:
#         models: a list of TransformerModel objects.

#     Returns:
#         A tuple (ids, scores), where ids is a Tensor with shape (batch_size, k,
#         max_seq_len) containing k translations for each input sentence in
#         model.inputs.x and scores is a Tensor with shape (batch_size, k)
#     """
#     assert len(models) == 1
#     model = models[0]
#     ids, scores = decode_greedy(model,
#                                 beam_size=beam_size,
#                                 normalization_alpha=normalization_alpha)
#     return ids, scores


# def decode_greedy(model, do_sample=False, beam_size=0,
#                   normalization_alpha=None):
#     # Determine size of current batch
#     batch_size, _ = get_shape_list(model.source_ids)
#     # Encode source sequences
#     with tf.name_scope('{:s}_encode'.format(model.name)):
#         enc_output, cross_attn_mask = model.enc.encode(model.source_ids,
#                                                        model.source_mask)

#     _, _, _, num_to_target = util.load_dictionaries(model.configs[0])
#     edges, labels = convert_text_to_graph(enc_output)
#     # Decode into target sequences
#     with tf.name_scope('{:s}_decode'.format(model.name)):
#         dec_output, scores = decode_at_test(model, model.dec, enc_output,
#             cross_attn_mask, edges, labels, batch_size, beam_size, do_sample, normalization_alpha)
#     return dec_output, scores


# def decode_at_test(model, decoder, enc_output, cross_attn_mask, edges, labels, batch_size, beam_size, do_sample, normalization_alpha):
#     """ Returns the probability distribution over target-side tokens conditioned on the output of the encoder;
#      performs decoding via auto-regression at test time. """

#     def _decode_step(target_embeddings, memories):
#         """ Decode the encoder-generated representations into target-side logits with auto-regression. """
#         # Propagate inputs through the encoder stack
#         dec_output = target_embeddings
#         # add gcn layers
#         if decoder.config.target_graph:
#             for layer_id in range(decoder.config.target_gcn_layers):
#                 orig_input = dec_output
#                 inputs = [dec_output, edges, labels]
#                 print_op = tf.Prinint([tf.shape(item) for item in inputs], "input shapes")

#                 with tf.control_dependencies([print_op]):
#                     dec_output = decoder.gcn_stack[layer_id].apply(inputs)
#         # NOTE: No self-attention mask is applied at decoding, as future information is unavailable
#         for layer_id in range(1, decoder.config.transformer_dec_depth + 1):
#             dec_output, memories['layer_{:d}'.format(layer_id)] = \
#                 decoder.decoder_stack[layer_id]['self_attn'].forward(
#                     dec_output, None, None, memories['layer_{:d}'.format(layer_id)])
#             dec_output, _ = \
#                 decoder.decoder_stack[layer_id]['cross_attn'].forward(dec_output, enc_output, cross_attn_mask)
#             dec_output = decoder.decoder_stack[layer_id]['ffn'].forward(dec_output)
#         # Return prediction at the final time-step to be consistent with the inference pipeline
#         dec_output = dec_output[:, -1, :]
#         return dec_output, memories

#     def _pre_process_targets(step_target_ids, current_time_step):
#         """ Pre-processes target token ids before they're passed on as input to the decoder
#         for auto-regressive decoding. """
#         # Embed target_ids
#         if decoder.config.target_graph:
#             step_target_ids, edge_labels, bias_labels = decoder._extract_labels(step_target_ids)
#             # use tf.py_func or tf.contrib.framework.py_func to process the target ids and dictionaries to the required format
#             raise NotImplementedError # TODO implement decoding for inference
#         target_embeddings = decoder._embed(step_target_ids)
#         signal_slice = positional_signal[:, current_time_step - 1: current_time_step, :]
#         target_embeddings += signal_slice
#         if decoder.config.transformer_dropout_embeddings > 0:
#             target_embeddings = tf.layers.dropout(target_embeddings,
#                                                   rate=decoder.config.transformer_dropout_embeddings, training=decoder.training)
#         return target_embeddings

#     def _decoding_function(step_target_ids, current_time_step, memories):
#         """ Generates logits for the target-side token predicted for the next-time step with auto-regression. """
#         # Embed the model's predictions up to the current time-step; add positional information, mask
#         target_embeddings = _pre_process_targets(step_target_ids, current_time_step)
#         # Pass encoder context and decoder embeddings through the decoder
#         dec_output, memories = _decode_step(target_embeddings, memories)
#         # Project decoder stack outputs and apply the soft-max non-linearity
#         step_logits = decoder.softmax_projection_layer.project(dec_output)
#         return step_logits, memories

#     with tf.variable_scope(decoder.name):
#         # Transpose encoder information in hybrid models
#         if decoder.from_rnn:
#             enc_output = tf.transpose(enc_output, [1, 0, 2])
#             cross_attn_mask = tf.transpose(cross_attn_mask, [3, 1, 2, 0])

#         positional_signal = get_positional_signal(decoder.config.translation_maxlen,
#                                                   decoder.config.embedding_size,
#                                                   decoder.float_dtype)

#         if decoder.config.target_graph:
#             raise NotImplementedError # TODO implement decoding for inference
#         # if decoder.config.target_graph:
#         #     # #     target_ids, edge_labels, bias_labels = _extract_labels(target_ids)
#         #     # edges_mask = get_target_edges_mask(tf.shape(target_ids)[-1], general_edge_mask)
#         #     # bias_mask = get_target_bias_mask(tf.shape(target_ids)[-1], general_bias_mask)
#         #     timestep = tf.shape(target_ids)[-1]
#         #     edges = get_tensor_from_times(timestep, edge_times)
#         #     labels = get_tensor_from_times(timestep, labels_times)

#         if beam_size > 0:
#             # Initialize target IDs with <GO>
#             initial_ids = tf.cast(tf.fill([batch_size], 1), dtype=decoder.int_dtype)
#             initial_memories = decoder._get_initial_memories(batch_size, beam_size=beam_size)
#             output_sequences, scores = _beam_search(_decoding_function,
#                                                    initial_ids,
#                                                    initial_memories,
#                                                    decoder.int_dtype,
#                                                    decoder.float_dtype,
#                                                    decoder.config.translation_maxlen,
#                                                    batch_size,
#                                                    beam_size,
#                                                    decoder.embedding_layer.get_vocab_size(),
#                                                    0,
#                                                    normalization_alpha)

#         else:
#             # Initialize target IDs with <GO>
#             initial_ids = tf.cast(tf.fill([batch_size, 1], 1), dtype=decoder.int_dtype)
#             initial_memories = decoder._get_initial_memories(batch_size, beam_size=1)
#             output_sequences, scores = greedy_search(model,
#                                                      _decoding_function,
#                                                      initial_ids,
#                                                      initial_memories,
#                                                      decoder.int_dtype,
#                                                      decoder.float_dtype,
#                                                      decoder.config.translation_maxlen,
#                                                      batch_size,
#                                                      0,
#                                                      do_sample,
#                                                      time_major=False)
#     return output_sequences, scores


# """ Inference functions for the transformer model. The generative process follows the 'Look, Generate, Update' paradigm, 
# as opposed to the 'Look, Update, Generate' paradigm employed by the deep-RNN. """


# # TODO: Add coverage penalty from Tu, Zhaopeng, et al.
# # TODO: "Modeling coverage for neural machine translation." arXiv preprint arXiv:1601.04811 (2016).

# # Note: Some inference mechanisms are adopted from the tensor2tensor library, with modifications

# # ============================================== Helper functions ==================================================== #

# def batch_to_beam(batch_tensor, beam_size):
#     """ Multiplies the batch tensor so as to match the size of the model's beam. """
#     batch_clones = [batch_tensor] * beam_size
#     new_beam = tf.stack(batch_clones, axis=1)
#     return new_beam


# def compute_batch_indices(batch_size, beam_size):
#     """ Generates a matrix of batch indices for the 'merged' beam tensor; each index denotes the batch from which the
#     sequence occupying the same relative position as the index within the 'merged' tensor belongs. """
#     batch_range = tf.range(batch_size * beam_size) // beam_size
#     batch_index_matrix = tf.reshape(batch_range, [batch_size, beam_size])
#     return batch_index_matrix


# def get_memory_invariants(memories):
#     """ Calculates the invariant shapes for the model memories (i.e. states of th RNN ar layer-wise attentions of the
#     transformer). """
#     memory_type = type(memories)
#     if memory_type == dict:
#         memory_invariants = dict()
#         for layer_id in memories.keys():
#             memory_invariants[layer_id] = {key: tf.TensorShape([None] * len(get_shape_list(memories[layer_id][key])))
#                                            for key in memories[layer_id].keys()}
#     else:
#         raise ValueError('Memory type not supported, must be a dictionary.')
#     return memory_invariants


# # Seems to work alright
# def gather_memories(memory_dict, gather_coordinates):
#     """ Gathers layer-wise memory tensors corresponding to top sequences from the provided memory dictionary
#     during beam search. """
#     # Initialize dicts
#     gathered_memories = dict()
#     # Get coordinate shapes
#     coords_dims = get_shape_list(gather_coordinates)

#     # Gather
#     for layer_key in memory_dict.keys():
#         layer_dict = memory_dict[layer_key]
#         gathered_memories[layer_key] = dict()

#         for attn_key in layer_dict.keys():
#             attn_tensor = layer_dict[attn_key]
#             attn_dims = get_shape_list(attn_tensor)
#             # Not sure if this is faster than the 'memory-less' version
#             flat_tensor = \
#                 tf.transpose(tf.reshape(attn_tensor, [-1, coords_dims[0]] + attn_dims[1:]), [1, 0, 2, 3])
#             gathered_values = tf.reshape(tf.transpose(tf.gather_nd(flat_tensor, gather_coordinates), [1, 0, 2, 3]),
#                                          [tf.multiply(coords_dims[1], coords_dims[0])] + attn_dims[1:])
#             gathered_memories[layer_key][attn_key] = gathered_values

#     return gathered_memories


# def gather_top_sequences(all_sequences,
#                          all_scores,
#                          all_scores_to_gather,
#                          all_eos_flags,
#                          all_memories,
#                          beam_size,
#                          batch_size,
#                          prefix):
#     """ Selects |beam size| sequences from a |beam size ** 2| sequence set; the selected sequences are used to update
#     sets of unfinished and completed decoding hypotheses. """
#     # Obtain indices of the top-k scores within the scores tensor
#     _, top_indices = tf.nn.top_k(all_scores, k=beam_size)
#     # Create a lookup-indices tensor for gathering the sequences associated with the top scores
#     batch_index_matrix = compute_batch_indices(batch_size, beam_size)
#     gather_coordinates = tf.stack([batch_index_matrix, top_indices], axis=2)  # coordinates in final dimension
#     # Collect top outputs
#     gathered_sequences = tf.gather_nd(all_sequences, gather_coordinates, name='{:s}_sequences'.format(prefix))
#     gathered_scores = tf.gather_nd(all_scores_to_gather, gather_coordinates, name='{:s}_scores'.format(prefix))
#     gathered_eos_flags = tf.gather_nd(all_eos_flags, gather_coordinates, name='{:s}_eos_flags'.format(prefix))
#     gathered_memories = None
#     if all_memories is not None:
#         gathered_memories = gather_memories(all_memories, gather_coordinates)
#         # gathered_memories = all_memories

#     return gathered_sequences, gathered_scores, gathered_eos_flags, gathered_memories


# # ============================================= Decoding functions =================================================== #


# def greedy_search(model,
#                   decoding_function,
#                   initial_ids,
#                   initial_memories,
#                   int_dtype,
#                   float_dtype,
#                   max_prediction_length,
#                   batch_size,
#                   eos_id,
#                   do_sample,
#                   time_major):
#     """ Greedily decodes the target sequence conditioned on the output of the encoder and the current output prefix. """

#     # Declare time-dimension
#     time_dim = int(not time_major)  # i.e. 0 if time_major, 1 if batch_major

#     # Define the 'body for the tf.while_loop() call
#     def _decoding_step(current_time_step, all_finished, next_ids, decoded_ids, decoded_score, memories):
#         """ Defines a single step of greedy decoding. """
#         # Propagate through decoder
#         step_logits, memories = decoding_function(next_ids, current_time_step, memories)
#         step_logits = model.sampling_utils.adjust_logits(step_logits)
#         # Calculate log probabilities for token prediction at current time-step
#         step_scores = tf.nn.log_softmax(step_logits)
#         # Determine next token to be generated, next_ids has shape [batch_size]
#         if do_sample:
#             next_ids = tf.squeeze(tf.multinomial(step_scores, num_samples=1, output_dtype=int_dtype), axis=1)
#         else:
#             # Greedy decoding
#             next_ids = tf.argmax(step_scores, -1, output_type=int_dtype)
#         # Collect scores associated with the selected tokens
#         score_coordinates = tf.stack([tf.range(batch_size, dtype=int_dtype), next_ids], axis=1)
#         decoded_score += tf.gather_nd(step_scores, score_coordinates)
#         # Concatenate newly decoded token ID with the previously decoded ones
#         decoded_ids = tf.concat([decoded_ids, tf.expand_dims(next_ids, 1)], 1)
#         # Extend next_id's dimensions to be compatible with input dimensionality for the subsequent step
#         next_ids = tf.expand_dims(next_ids, time_dim)
#         # Check if generation has concluded with <EOS>
#         # all_finished |= tf.equal(tf.squeeze(next_ids, axis=time_dim), eos_id)
#         all_finished |= tf.equal(tf.reduce_prod(decoded_ids - eos_id, axis=time_dim), eos_id)

#         return current_time_step + 1, all_finished, next_ids, decoded_ids, decoded_score, memories

#     # Define the termination condition for the tf.while_loop() call
#     def _continue_decoding(_current_time_step, _all_finished, *_):
#         """ Returns 'False' if all of the sequences in the generated sequence batch exceeded the maximum specified
#         length or terminated with <EOS>, upon which the while loop is exited. """
#         continuation_check = \
#             tf.logical_and(tf.less(_current_time_step, max_prediction_length),
#                            tf.logical_not(tf.reduce_all(_all_finished)))

#         return continuation_check

#     # Initialize decoding-relevant variables and containers
#     current_time_step = tf.constant(1)
#     all_finished = tf.fill([batch_size], False)  # None of the sequences is marked as finished
#     next_ids = initial_ids
#     decoded_ids = tf.zeros([batch_size, 0], dtype=int_dtype)  # Sequence buffer is empty
#     decoded_score = tf.zeros([batch_size], dtype=float_dtype)
#     memories = initial_memories

#     # Execute the auto-regressive decoding step via while loop
#     _, _, _, decoded_ids, log_scores, memories = \
#         tf.while_loop(cond=_continue_decoding,
#                       body=_decoding_step,
#                       loop_vars=[current_time_step, all_finished, next_ids, decoded_ids, decoded_score, memories],
#                       shape_invariants=[tf.TensorShape([]),
#                                         tf.TensorShape([None]),
#                                         tf.TensorShape([None, None]),
#                                         tf.TensorShape([None, None]),
#                                         tf.TensorShape([None]),
#                                         get_memory_invariants(memories)],
#                       parallel_iterations=10,
#                       swap_memory=False,
#                       back_prop=False)

#     # Should return logits also, for training
#     return decoded_ids, log_scores


# def _beam_search(decoding_function,
#                 initial_ids,
#                 initial_memories,
#                 int_dtype,
#                 float_dtype,
#                 translation_maxlen,
#                 batch_size,
#                 beam_size,
#                 vocab_size,
#                 eos_id,
#                 normalization_alpha):
#     """ Decodes the target sequence by maintaining a beam of candidate hypotheses, thus allowing for better exploration
#     of the hypothesis space; optionally applies scaled length normalization; based on the T2T implementation.

#         alive = set of n unfinished hypotheses presently within the beam; n == beam_size
#         finished = set of n finished hypotheses, each terminating in <EOS>; n == beam_size

#     """

#     def _extend_hypotheses(current_time_step, alive_sequences, alive_log_probs, alive_memories):
#         """ Generates top-k extensions of the alive beam candidates from the previous time-step, which are subsequently
#         used to update the alive and finished sets at the current time-step; top-k = 2 s* beam_size """
#         # Get logits for the current prediction step
#         next_ids = alive_sequences[:, :, -1]  # [batch_size, beam_size]
#         next_ids = tf.transpose(next_ids, [1, 0])  # [beam_size, batch_size]; transpose to match model
#         next_ids = tf.reshape(next_ids, [-1, 1])  # [beam_size * batch_size, 1]

#         step_logits, alive_memories = decoding_function(next_ids, current_time_step, alive_memories)
#         step_logits = tf.reshape(step_logits, [beam_size, batch_size, -1])  # [beam_size, batch_size, num_words]
#         step_logits = tf.transpose(step_logits, [1, 0, 2])  # [batch_size, beam_size, num_words]; transpose back

#         # Calculate the scores for all possible extensions of alive hypotheses
#         candidate_log_probs = tf.nn.log_softmax(step_logits, axis=-1)
#         curr_log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

#         # Apply length normalization
#         length_penalty = 1.
#         if normalization_alpha > 0.:
#             length_penalty = ((5. + tf.to_float(current_time_step)) ** normalization_alpha) / \
#                              ((5. + 1.) ** normalization_alpha)
#         curr_scores = curr_log_probs / length_penalty

#         # at first time step, all beams are identical - pick first
#         # at other time steps, flatten all beams
#         flat_curr_scores = tf.cond(tf.equal(current_time_step, 1),
#                       lambda: curr_scores[:,0],
#                       lambda: tf.reshape(curr_scores, [batch_size, -1]))

#         # Select top-k highest scores
#         top_scores, top_ids = tf.nn.top_k(flat_curr_scores, k=beam_size)

#         # Recover non-normalized scores for tracking
#         top_log_probs = top_scores * length_penalty

#         # Determine the beam from which the top-scoring items originate and their identity (i.e. token-ID)
#         top_beam_indices = top_ids // vocab_size
#         top_ids %= vocab_size

#         # Determine the location of top candidates
#         batch_index_matrix = compute_batch_indices(batch_size, beam_size)  # [batch_size, beam_size]
#         top_coordinates = tf.stack([batch_index_matrix, top_beam_indices], axis=2)

#         # Extract top decoded sequences
#         top_sequences = tf.gather_nd(alive_sequences, top_coordinates)  # [batch_size, beam_size, sent_len]
#         top_sequences = tf.concat([top_sequences, tf.expand_dims(top_ids, axis=2)], axis=2)

#         # Extract top memories
#         top_memories = gather_memories(alive_memories, top_coordinates)
#         # top_memories = alive_memories

#         # Check how many of the top sequences have terminated
#         top_eos_flags = tf.equal(top_ids, eos_id)  # [batch_size, beam_size]

#         return top_sequences, top_log_probs, top_scores, top_eos_flags, top_memories

#     def _update_alive(top_sequences, top_scores, top_log_probs, top_eos_flags, top_memories):
#         """ Assembles an updated set of unfinished beam candidates from the set of top-k translation hypotheses
#         generated at the current time-step; top-k for the incoming sequences in 2 * beam_size """
#         # Exclude completed sequences from the alive beam by setting their scores to a large negative value
#         top_scores += tf.to_float(top_eos_flags) * (-1. * 1e7)
#         # Update the alive beam
#         updated_alive_sequences, updated_alive_log_probs, updated_alive_eos_flags, updated_alive_memories = \
#             gather_top_sequences(top_sequences,
#                                  top_scores,
#                                  top_log_probs,
#                                  top_eos_flags,
#                                  top_memories,
#                                  beam_size,
#                                  batch_size,
#                                  'alive')

#         return updated_alive_sequences, updated_alive_log_probs, updated_alive_eos_flags, updated_alive_memories

#     def _update_finished(finished_sequences, finished_scores, finished_eos_flags, top_sequences, top_scores,
#                          top_eos_flags):
#         """ Updates the list of completed translation hypotheses (i.e. ones terminating in <EOS>) on the basis of the
#         top-k hypotheses generated at the current time-step; top-k for the incoming sequences in 2 * beam_size """
#         # Match the length of the 'finished sequences' tensor with the length of the 'finished scores' tensor
#         zero_padding = tf.zeros([batch_size, beam_size, 1], dtype=int_dtype)
#         finished_sequences = tf.concat([finished_sequences, zero_padding], axis=2)
#         # Exclude incomplete sequences from the finished beam by setting their scores to a large negative value
#         top_scores += (1. - tf.to_float(top_eos_flags)) * (-1. * 1e7)
#         # Combine sequences finished at previous time steps with the top sequences from current time step, as well as
#         # their scores and eos-flags, for the selection of a new, most likely, set of finished sequences
#         top_finished_sequences = tf.concat([finished_sequences, top_sequences], axis=1)
#         top_finished_scores = tf.concat([finished_scores, top_scores], axis=1)
#         top_finished_eos_flags = tf.concat([finished_eos_flags, top_eos_flags], axis=1)
#         # Update the finished beam
#         updated_finished_sequences, updated_finished_scores, updated_finished_eos_flags, _ = \
#             gather_top_sequences(top_finished_sequences,
#                                  top_finished_scores,
#                                  top_finished_scores,
#                                  top_finished_eos_flags,
#                                  None,
#                                  beam_size,
#                                  batch_size,
#                                  'finished')

#         return updated_finished_sequences, updated_finished_scores, updated_finished_eos_flags

#     def _decoding_step(current_time_step,
#                        alive_sequences,
#                        alive_log_probs,
#                        finished_sequences,
#                        finished_scores,
#                        finished_eos_flags,
#                        alive_memories):
#         """ Defines a single step of greedy decoding. """
#         # 1. Get the top sequences/ scores/ flags for the current time step
#         top_sequences, top_log_probs, top_scores, top_eos_flags, top_memories = \
#             _extend_hypotheses(current_time_step,
#                                alive_sequences,
#                                alive_log_probs,
#                                alive_memories)

#         # 2. Update the alive beam
#         alive_sequences, alive_log_probs, alive_eos_flags, alive_memories = \
#             _update_alive(top_sequences,
#                           top_scores,
#                           top_log_probs,
#                           top_eos_flags,
#                           top_memories)

#         # 3. Update the finished beam
#         finished_sequences, finished_scores, finished_eos_flags = \
#             _update_finished(finished_sequences,
#                              finished_scores,
#                              finished_eos_flags,
#                              top_sequences,
#                              top_scores,
#                              top_eos_flags)

#         return current_time_step + 1, alive_sequences, alive_log_probs, finished_sequences, finished_scores, \
#                finished_eos_flags, alive_memories

#     def _continue_decoding(curr_time_step,
#                            alive_sequences,
#                            alive_log_probs,
#                            finished_sequences,
#                            finished_scores,
#                            finished_eos_flags,
#                            alive_memories):
#         """ Returns 'False' if all of the sequences in the extended hypotheses exceeded the maximum specified
#         length or if none of the extended hypotheses are more likely than the lowest scoring finished hypothesis. """
#         # Check if the maximum prediction length has been reached
#         length_criterion = tf.less(curr_time_step, translation_maxlen)

#         # Otherwise, check if the most likely alive hypothesis is less likely than the least probable completed sequence
#         # Calculate the best possible score of the most probably sequence currently alive
#         max_length_penalty = 1.
#         if normalization_alpha > 0.:
#             max_length_penalty = ((5. + tf.to_float(translation_maxlen)) ** normalization_alpha) / \
#                                  ((5. + 1.) ** normalization_alpha)

#         highest_alive_score = alive_log_probs[:, 0] / max_length_penalty
#         # Calculate the score of the least likely sequence currently finished
#         lowest_finished_score = tf.reduce_min(finished_scores * tf.cast(finished_eos_flags, float_dtype), axis=1)
#         # Account for the case in which none of the sequences in 'finished' have terminated so far;
#         # In that case, each of the unfinished sequences is assigned a high negative probability, so that the
#         # termination condition is not met
#         mask_unfinished = (1. - tf.to_float(tf.reduce_any(finished_eos_flags, 1))) * (-1. * 1e7)
#         lowest_finished_score += mask_unfinished

#         # Check is the current highest alive score is lower than the current lowest finished score
#         likelihood_criterion = tf.logical_not(tf.reduce_all(tf.greater(lowest_finished_score, highest_alive_score)))

#         # Decide whether to continue the decoding process
#         do_continue = tf.logical_and(length_criterion, likelihood_criterion)
#         return do_continue

#     # Initialize alive sequence and score trackers and expand to beam size
#     alive_log_probs = tf.zeros([batch_size, beam_size])

#     # Initialize decoded sequences
#     alive_sequences = tf.expand_dims(batch_to_beam(initial_ids, beam_size), 2)

#     # Initialize finished sequence, score, and flag trackers
#     finished_sequences = tf.expand_dims(batch_to_beam(initial_ids, beam_size), 2)
#     finished_scores = tf.ones([batch_size, beam_size]) * (-1. * 1e7)  # initialize to a low value
#     finished_eos_flags = tf.zeros([batch_size, beam_size], dtype=tf.bool)

#     # Initialize memories
#     alive_memories = initial_memories

#     # Execute the auto-regressive decoding step via while loop
#     _, alive_sequences, alive_log_probs, finished_sequences, finished_scores, finished_eos_flags, _ = \
#         tf.while_loop(
#             _continue_decoding,
#             _decoding_step,
#             [tf.constant(1), alive_sequences, alive_log_probs, finished_sequences, finished_scores, finished_eos_flags,
#              alive_memories],
#             shape_invariants=[tf.TensorShape([]),
#                               tf.TensorShape([None, None, None]),
#                               alive_log_probs.get_shape(),
#                               tf.TensorShape([None, None, None]),
#                               finished_scores.get_shape(),
#                               finished_eos_flags.get_shape(),
#                               get_memory_invariants(alive_memories)],
#             parallel_iterations=10,
#             swap_memory=False,
#             back_prop=False)

#     alive_sequences.set_shape((None, beam_size, None))
#     finished_sequences.set_shape((None, beam_size, None))

#     # Account for the case in which a particular sequence never terminates in <EOS>;
#     # in that case, copy the contents of the alive beam for that item into the finished beam (sequence + score)
#     # tf.reduce_any(finished_eos_flags, 1) is False if there exists no completed translation hypothesis for a source
#     # sentence in either of the beams , i.e. no replacement takes place if there is at least one finished translation
#     finished_sequences = tf.where(tf.reduce_any(finished_eos_flags, 1), finished_sequences, alive_sequences)
#     # Attention: alive_scores are not length normalized!
#     finished_scores = tf.where(tf.reduce_any(finished_eos_flags, 1), finished_scores, alive_log_probs)
#     # Truncate initial <GO> in finished sequences
#     finished_sequences = finished_sequences[:, :, 1:]

#     return finished_sequences, finished_scores
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

        def _decoding_function(step_target_ids, current_time_step, memories):
            """Single-step decoding function.

            Args:
                step_target_ids: Tensor with shape (batch_size)
                current_time_step: scalar Tensor.
                memories: dictionary (see top-level class description)

            Returns:
            """
            with tf.name_scope(self._scope):
                # TODO Is this necessary?
                vocab_ids = tf.reshape(step_target_ids, [-1, 1])
                # Look up embeddings for target IDs.
                target_embeddings = decoder._embed(vocab_ids)

                # Add positional signal.
                signal_slice = positional_signal[
                    :, current_time_step-1:current_time_step, :]
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
                        edges, labels = tf.compat.v1.py_func(self.extract_graph, [step_target_ids], [tf.float32, tf.float32], stateful=False) # this or perhaps the more efficient tf based .compat.v1.py_function?
                        # edges, labels = tf.numpy_function(self.extract_graph, [step_target_ids], [tf.float32, tf.float32]) # this or perhaps the more efficient tf based .compat.v1.py_function?
                        # edges = tf.reshape(edges, [self.config.batch_size, self.config.target_vocab_size, self.config.target_vocab_size, 3])
                        edges.set_shape([None, self.config.maxlen, self.config.maxlen, 3])
                        labels.set_shape([None, self.config.maxlen, self.config.maxlen, self.config.target_labels_num])
                        tf.ensure_shape(edges, [None, self.config.maxlen, self.config.maxlen, 3])
                        tf.ensure_shape(labels, [None, self.config.maxlen, self.config.maxlen, self.config.target_labels_num])
                        edges = util.dense_to_sparse_tensor(edges)
                        labels = util.dense_to_sparse_tensor(labels)
                        print(edges.get_shape())
                        printops = []
                        printops.append(tf.Print([], [edges.indices, edges.values], "pythoned edges", 10, 50))
                        with tf.control_dependencies(printops):
                            inputs = [layer_output, edges, labels]
                        layer_output = self.model.dec.gcn_stack[layer_id].apply(inputs)
                # Propagate values through the decoder stack.
                # NOTE: No self-attention mask is applied at decoding, as
                #       future information is unavailable.
                for layer_id in range(1, self.config.transformer_dec_depth+1):
                    layer = decoder.decoder_stack[layer_id]
                    mem_key = 'layer_{:d}'.format(layer_id)
                    layer_output, memories[mem_key] = \
                        layer['self_attn'].forward(
                            layer_output, None, None, memories[mem_key])
                    layer_output, _ = layer['cross_attn'].forward(
                        layer_output, encoder_output.enc_output,
                        encoder_output.cross_attn_mask)
                    layer_output = layer['ffn'].forward(layer_output)
                # Return prediction at the final time-step to be consistent
                # with the inference pipeline.
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
                memories['layer_{:d}'.format(layer_id)] = { \
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
                        [None]*len(tf_utils.get_shape_list(layer_mems[key])))
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

            shapes = { gather_coordinates: ('batch_size_x', 'beam_size', 2) }
            tf_utils.assert_shapes(shapes)

            coords_shape = tf.shape(gather_coordinates)
            batch_size_x, beam_size = coords_shape[0], coords_shape[1]

            def gather_attn(attn):
                # TODO Specify second and third?
                shapes = { attn: ('batch_size', None, None) }
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
        label_dict = self.model.target_labels_dict
        inv_dict = {v: k for k, v in label_dict.items()}
        strs = [[inv_dict[idn] for idn in row] for row in ids] #TODO is this indeed the right format of all the inputs?
        return convert_text_to_graph(strs, self._config.maxlen, label_dict, self.model.target_labels_num, True)