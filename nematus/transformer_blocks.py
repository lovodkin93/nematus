"""Adapted from Nematode: https://github.com/demelin/nematode """

import sys
import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .transformer_attention_modules import MultiHeadAttentionLayer
    from .transformer_layers import \
        ProcessingLayer, \
        FeedForwardNetwork

except (ModuleNotFoundError, ImportError) as e:
    from transformer_attention_modules import MultiHeadAttentionLayer
    from transformer_layers import \
        ProcessingLayer, \
        FeedForwardNetwork

# from attention_modules import SingleHeadAttentionLayer, FineGrainedAttentionLayer


class AttentionBlock(object):
    """ Defines a single attention block (referred to as 'sub-layer' in the paper) comprising of a single multi-head
    attention layer preceded by a pre-processing layer and followed by a post-processing layer. """

    def __init__(self,
                 config,
                 float_dtype,
                 self_attention,
                 training,
                 isDecoder,
                 layer_id,
                 from_rnn=False,
                 tie_attention=False):
        # Set attributes
        self.self_attention = self_attention
        if not tie_attention:
            if self_attention:
                attn_name = 'self_attn'
            else:
                attn_name = 'cross_attn'
        else:
            attn_name = 'tied_attn'

        memory_size = config.state_size
        if from_rnn:
            memory_size *= 2

        # Build layers
        self.pre_attn = ProcessingLayer(config.state_size,
                                        use_layer_norm=True,
                                        dropout_rate=0.,
                                        training=training,
                                        name='pre_{:s}_sublayer'.format(attn_name))

        self.attn = MultiHeadAttentionLayer(memory_size,
                                            config.state_size,
                                            config.state_size,
                                            config.state_size,
                                            config.state_size,
                                            config.transformer_num_heads,
                                            float_dtype,
                                            dropout_attn=config.transformer_dropout_attn,
                                            training=training,
                                            config=config,
                                            isDecoder=isDecoder,
                                            layer_id=layer_id,
                                            name='{:s}_sublayer'.format(attn_name))

        self.post_attn = ProcessingLayer(config.state_size,
                                         use_layer_norm=False,
                                         dropout_rate=config.transformer_dropout_residual,
                                         training=training,
                                         name='post_{:s}_sublayer'.format(attn_name))

    def forward(self, inputs, memory_context, attn_mask, pre_softmax_scaled_attn_mask,post_softmax_scaled_attn_mask, cross_attention_keys_update_rules=None, target_mask_learning=None, layer_memories=None, isDecoder=False, isInference=False): # make sure everyone calling sends pre_softmax_scaled_attn_mask
        """ Propagates input data through the block. """
        if not self.self_attention:
            assert (memory_context is not None), \
                'Encoder memories have to be provided for encoder-decoder attention computation.'

        ################################ PRINT ###################################################
        # print_ops = []
        # print_ops.append(tf.compat.v1.Print([], [tf.shape(inputs)], "AVIVSL6: attn_mask: ", summarize=10000))
        # # if not isDecoder:
        # #     if target_mask_learning_input_list is None:
        # #         print_ops.append(tf.compat.v1.Print([], [], "AVIVSL6: GOOD! no mask in encoder", summarize=10000))
        # #     else:
        # #         print_ops.append(tf.compat.v1.Print([], [], "AVIVSL6: BAD! mask in encoder", summarize=10000))
        # if isDecoder:
        #     if attn_mask is None:
        #         name = 'self_attn' if self.self_attention else 'cross_attn'
        #         train_or_inference = 'inference' if isInference else 'train'
        #         print_ops.append(tf.compat.v1.Print([], [], "AVIVSL6: No mask in " + name + ' in ' +  train_or_inference, summarize=10000))
        #     else:
        #         name = 'self_attn' if self.self_attention else 'cross_attn'
        #         train_or_inference = 'inference' if isInference else 'train'
        #         print_ops.append(tf.compat.v1.Print([], [tf.shape(attn_mask)], "AVIVSL6: attn_mask in " + name + ' in ' +  train_or_inference, summarize=10000))
        # with tf.control_dependencies(print_ops):
        #     inputs = 1 * inputs
        #############################################################################################


        attn_inputs = self.pre_attn.forward(inputs)
        attn_outputs, layer_memories, attn_softmax_weights, target_same_scene_learnt_mask = self.attn.forward(attn_inputs, memory_context, attn_mask, pre_softmax_scaled_attn_mask,post_softmax_scaled_attn_mask, layer_memories, cross_attention_keys_update_rules=cross_attention_keys_update_rules, target_mask_learning=target_mask_learning, isDecoder=isDecoder, isInference=isInference) #goes to MultiHeadAttentionLayer's forward in nematus.transformer_attention_modules
        block_out = self.post_attn.forward(attn_outputs, residual_inputs=inputs)

        ############################################### PRINTING #######################################################
        # printops = []
        # if isDecoder and not isInference:
        #     printops.append(tf.compat.v1.Print([], [tf.shape(target_mask_learning_input_list[0])],
        #                                    "AVIVSL8: target_mask_learning_input_list ", summarize=10000))
        #     # printops.append(tf.compat.v1.Print([], [tf.shape(layer_memories), layer_memories],
        #     #                                "AVIVSL8: attn_softmax_weights ", summarize=10000))
        # with tf.control_dependencies(printops):
        #     block_out = block_out * 1
        ################################################################################################################

        return block_out, layer_memories, attn_softmax_weights, target_same_scene_learnt_mask


class FFNBlock(object):
    """ Defines a single feed-forward network block (referred to as 'sub-layer' in the transformer paper) comprising of
    a single feed-forward network preceded by a pre-processing layer and followed by a post-processing layer. """

    def __init__(self,
                 config,
                 ffn_dims,
                 float_dtype,
                 is_final,
                 training):
        # Set attributes
        self.is_final = is_final

        # Build layers
        self.pre_ffn = ProcessingLayer(config.state_size,
                                       use_layer_norm=True,
                                       dropout_rate=0.,
                                       training=training,
                                       name='pre_ffn_sublayer')
        self.ffn = FeedForwardNetwork(ffn_dims,
                                      float_dtype,
                                      use_bias=True,
                                      activation=tf.nn.relu,
                                      use_layer_norm=False,
                                      dropout_rate=config.transformer_dropout_relu,
                                      training=training,
                                      name='ffn_sublayer')
        self.post_ffn = ProcessingLayer(config.state_size,
                                        use_layer_norm=False,
                                        dropout_rate=config.transformer_dropout_residual,
                                        training=training,
                                        name='post_ffn_sublayer')
        if is_final:
            self.pre_final = ProcessingLayer(config.state_size,
                                             use_layer_norm=True,
                                             dropout_rate=0.,
                                             training=training,
                                             name='final_transform')

    def forward(self, inputs):
        """ Propagates input data through the block. """
        ffn_inputs = self.pre_ffn.forward(inputs)
        ffn_outputs = self.ffn.forward(ffn_inputs)
        block_out = self.post_ffn.forward(ffn_outputs, residual_inputs=inputs)
        if self.is_final:
            block_out = self.pre_final.forward(block_out)
        return block_out
