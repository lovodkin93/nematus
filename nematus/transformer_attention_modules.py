"""Adapted from Nematode: https://github.com/demelin/nematode """

import sys
import ast
import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .tf_utils import get_shape_list
    from .transformer_layers import FeedForwardLayer, matmul_nd
except (ModuleNotFoundError, ImportError) as e:
    from tf_utils import get_shape_list
    from transformer_layers import FeedForwardLayer, matmul_nd


class MultiHeadAttentionLayer(object):
    """ Defines the multi-head, multiplicative attention mechanism;
    based on the tensor2tensor library implementation. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 total_key_dims,
                 total_value_dims,
                 output_dims,
                 num_heads,
                 float_dtype,
                 dropout_attn,
                 training,
                 config,
                 isDecoder,
                 layer_id,
                 name=None):

        # Set attributes
        self.reference_dims = reference_dims
        self.hypothesis_dims = hypothesis_dims
        self.total_key_dims = total_key_dims
        self.total_value_dims = total_value_dims
        self.output_dims = output_dims
        self.num_heads = num_heads
        self.float_dtype = float_dtype
        self.dropout_attn = dropout_attn
        self.training = training
        self.config = config
        self.name = name
        self.isDecoder = isDecoder
        self.layer_id=layer_id

        # Check if the specified hyper-parameters are consistent
        if total_key_dims % num_heads != 0:
            raise ValueError('Specified total attention key dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_key_dims, num_heads))
        if total_value_dims % num_heads != 0:
            raise ValueError('Specified total attention value dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_value_dims, num_heads))

        # dims of FC FFN to learn the target same scene mask
        if self.config.target_same_scene_masks_FC_FFN_how == 'just_source_mask':
            self.target_same_scene_mask_projection_in_size = max(self.config.maxlen, self.config.translation_maxlen) ** 2
        else:
            self.target_same_scene_mask_projection_in_size = max(self.config.maxlen + 1,self.config.translation_maxlen + 1) ** 2 + self.config.embedding_size
        self.target_same_scene_mask_projection_out_size = max(self.config.maxlen + 1, self.config.translation_maxlen + 1)

        # which layers in decoder to add the FC_FFN learning the target same_scene_mask
        if self.config.target_same_scene_masks_FC_FFN_layers == 'all_layers':
            self.target_same_scene_mask_projection_layers = [layer for layer in range(1,self.config.transformer_dec_depth+1)]
        else:
            self.target_same_scene_mask_projection_layers = ast.literal_eval(self.config.target_same_scene_masks_FC_FFN_layers)

        # Instantiate parameters
        with tf.compat.v1.variable_scope(self.name):
            self.queries_projection = FeedForwardLayer(self.hypothesis_dims,
                                                       self.total_key_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='queries_projection')

            self.keys_projection = FeedForwardLayer(self.reference_dims,
                                                    self.total_key_dims,
                                                    float_dtype,
                                                    dropout_rate=0.,
                                                    activation=None,
                                                    use_bias=False,
                                                    use_layer_norm=False,
                                                    training=self.training,
                                                    name='keys_projection')

            self.values_projection = FeedForwardLayer(self.reference_dims,
                                                      self.total_value_dims,
                                                      float_dtype,
                                                      dropout_rate=0.,
                                                      activation=None,
                                                      use_bias=False,
                                                      use_layer_norm=False,
                                                      training=self.training,
                                                      name='values_projection')

            self.context_projection = FeedForwardLayer(self.total_value_dims,
                                                       self.output_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='context_projection')


            if self.config.target_same_scene_head_FC_FFN and self.name=='self_attn_sublayer' and self.isDecoder and self.layer_id in self.target_same_scene_mask_projection_layers:
                self.target_same_scene_mask_projection = FeedForwardLayer(self.target_same_scene_mask_projection_in_size,
                                                           self.target_same_scene_mask_projection_out_size,
                                                           float_dtype,
                                                           dropout_rate=0.,
                                                           activation=None,
                                                           use_bias=False,
                                                           use_layer_norm=False,
                                                           training=self.training,
                                                           name='target_same_scene_mask_projection')



    def _compute_attn_inputs(self, query_context, memory_context):
        """ Computes query, key, and value tensors used by the attention function for the calculation of the
        time-dependent context representation. """
        queries = self.queries_projection.forward(query_context)
        keys = self.keys_projection.forward(memory_context)
        values = self.values_projection.forward(memory_context)
        return queries, keys, values

    def _split_among_heads(self, inputs):
        """ Splits the attention inputs among multiple heads. """
        # Retrieve the depth of the input tensor to be split (input is 3d)
        inputs_dims = get_shape_list(inputs)
        inputs_depth = inputs_dims[-1]

        # Assert the depth is compatible with the specified number of attention heads
        if isinstance(inputs_depth, int) and isinstance(self.num_heads, int):
            assert inputs_depth % self.num_heads == 0, \
                ('Attention inputs depth {:d} is not evenly divisible by the specified number of attention heads {:d}'
                 .format(inputs_depth, self.num_heads))
        split_inputs = tf.reshape(inputs, inputs_dims[:-1] + [self.num_heads, inputs_depth // self.num_heads])
        return split_inputs

    def _merge_from_heads(self, split_inputs):
        """ Inverts the _split_among_heads operation. """
        # Transpose split_inputs to perform the merge along the last two dimensions of the split input
        split_inputs = tf.transpose(a=split_inputs, perm=[0, 2, 1, 3])
        # Retrieve the depth of the tensor to be merged
        split_inputs_dims = get_shape_list(split_inputs)
        split_inputs_depth = split_inputs_dims[-1]
        # Merge the depth and num_heads dimensions of split_inputs
        merged_inputs = tf.reshape(split_inputs, split_inputs_dims[:-2] + [self.num_heads * split_inputs_depth])
        return merged_inputs

    def _dot_product_attn(self, queries, keys, values, attn_mask, pre_softmax_scaled_attn_mask,post_softmax_scaled_attn_mask, target_mask_learning, scaling_on, isDecoder, isInference): #TODO: AVIVSL: delete the isDecoder in the end
        """ Defines the dot-product attention function; see Vaswani et al.(2017), Eq.(1). """
        # query/ key/ value have shape = [batch_size, time_steps, num_heads, num_features]
        # Tile keys and values tensors to match the number of decoding beams; ignored if already done by fusion module
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]

        # ################################################## PRINTS ########################################################
        # print_ops = []
        # if isDecoder and isInference:

            # print_ops.append(
            #                 tf.compat.v1.Print([], [], "Inference2! in:" + self.name, 50, 100))
            # if attn_mask is not None:
            #     print_ops.append(
            #         tf.compat.v1.Print([], [tf.shape(attn_mask)], "AVIVSL8: attn_mask in inference", summarize=10000))
            # else:
            #     print_ops.append(
            #         tf.compat.v1.Print([], [], "No mask in inference!!!!", summarize=10000))
        # # if self.name == "cross_attn_sublayer":
        # #     print_ops.append(
        # #         tf.compat.v1.Print([], [tf.shape(attn_mask), attn_mask], "AVIVSL6: attn_mask for encoder is" + self.name, summarize=10000))
        # # if isDecoder and self.name == "self_attn_sublayer":
        # #     print_ops.append(
        # #         tf.compat.v1.Print([], [tf.shape(attn_mask), attn_mask], "AVIVSL6: attn_mask for decoder is" + self.name, summarize=10000))
        # #     print_ops.append(tf.compat.v1.Print([], [tf.shape(values), values[0, 0, :, :10]], "in_values" + self.name, 50, 100))
        # #     print_ops.append(tf.compat.v1.Print([], [tf.shape(queries), queries[0, 0, :, :10]], "in_queries " + self.name, 50, 100))
        # #     print_ops.append(tf.compat.v1.Print([], [tf.shape(keys), keys[0, 0, :, :10]], "in_keys" + self.name, 50, 100))
        # # if "self" in self.name:
        # #     print_ops = []
        # with tf.control_dependencies(print_ops):
        #     num_beams = num_beams * 1
        # #################################################################################################################
        keys = tf.cond(pred=tf.greater(num_beams, 1), true_fn=lambda: tf.tile(keys, [num_beams, 1, 1, 1]),
                       false_fn=lambda: keys)
        values = tf.cond(pred=tf.greater(num_beams, 1), true_fn=lambda: tf.tile(values, [num_beams, 1, 1, 1]),
                         false_fn=lambda: values)


        ################################################## PRINTS ########################################################
        # print_ops = []
        # print_ops.append(
        #     tf.compat.v1.Print([], [tf.shape(keys), tf.shape(values)], "AVIVSL6: keys+values shape is " + self.name, summarize=10000))
        # if self.name == "cross_attn_sublayer":
        #     print_ops.append(
        #         tf.compat.v1.Print([], [tf.shape(keys), keys[:,:,0,:]], "AVIVSL6: keys size in " + self.name, summarize=10000))
        #     print_ops.append(
        #         tf.compat.v1.Print([], [tf.shape(values), values[:,:,0,:]], "AVIVSL6: values size in " + self.name, summarize=10000))
        #     print_ops.append(
        #         tf.compat.v1.Print([], [tf.shape(queries)], "AVIVSL6: queries size in " + self.name, summarize=10000))
        # with tf.control_dependencies(print_ops):
        #     num_beams = num_beams * 1
        #################################################################################################################



        # print_ops = []
        # print_ops.append(
        #     tf.compat.v1.Print([], [tf.shape(values),tf.shape(keys)],"tiled_values == tiled_keys? " + self.name, 50, 100))
        # with tf.control_dependencies(print_ops):



        # Transpose split inputs
        queries = tf.transpose(a=queries, perm=[0, 2, 1, 3])
        values = tf.transpose(a=values, perm=[0, 2, 1, 3])
        attn_logits = tf.matmul(queries, tf.transpose(a=keys, perm=[0, 2, 3, 1]))



        ################################################## PRINTS ########################################################
        # print_ops = []
        # if self.name == 'self_attn_sublayer' and not isDecoder:
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[0,0,tf.shape(attn_logits)[3]-8:,:]], "AVIVSL8: attn_logits[0,0,:,:] " + self.name, summarize=12000))
        # if isDecoder:
        #     if isInference:
        #         if attn_mask is not None:
        #             print_ops.append(
        #                 tf.compat.v1.Print([], [tf.shape(attn_mask), attn_mask], "AVIVSL8: attn_mask in inference in " + self.name, summarize=10000))
        #         else:
        #             print_ops.append(
        #                 tf.compat.v1.Print([], [], "AVIVSL8: No attn_mask in inference in " + self.name, summarize=10000))
        #     else:
        #         if attn_mask is not None:
        #             print_ops.append(
        #                 tf.compat.v1.Print([], [tf.shape(attn_mask), attn_mask], "AVIVSL8: attn_mask in train in " + self.name, summarize=10000))
        #         else:
        #             print_ops.append(
        #                 tf.compat.v1.Print([], [], "AVIVSL8: No attn_mask in train in " + self.name, summarize=10000))
        # with tf.control_dependencies(print_ops):
        #     attn_logits = attn_logits * 1
        #################################################################################################################
        ################################################## PRINTS ########################################################
        # print_ops = []
        # if self.name == "cross_attn_sublayer":
        #      print_ops.append(
        #          tf.compat.v1.Print([], [tf.shape(attn_logits)], "AVIVSL6: attn_logits size" + self.name, summarize=10000))
        # # if isDecoder and self.name == "self_attn_sublayer":
        # #     print_ops.append(
        # #         tf.compat.v1.Print([], [tf.shape(attn_mask), tf.shape(attn_logits)], "AVIVSL6: attn_mask and attn_logits for decoder is" + self.name, summarize=10000))
        # with tf.control_dependencies(print_ops):
        #      attn_logits = attn_logits * 1
        #################################################################################################################

        # Scale attention_logits by key dimensions to prevent softmax saturation, if specified
        if scaling_on:
            key_dims = get_shape_list(keys)[-1]
            normalizer = tf.sqrt(tf.cast(key_dims, self.float_dtype))
            attn_logits /= normalizer




        ################################################## PRINTS ########################################################
        # print_ops = []
        # if isDecoder and 'self_attn' in self.name:
        #     inference_or_train = 'inference' if isInference else 'train'
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(attn_logits)], "AVIVSL6: attn_logits size" + self.name + ' in ' + inference_or_train, summarize=10000))
        # with tf.control_dependencies(print_ops):
        #     attn_logits = attn_logits * 1
        #################################################################################################################






        # Optionally mask out positions which should not be attended to
        # attention mask should have shape=[batch, num_heads, query_length, key_length]
        # attn_logits has shape=[batch, num_heads, query_length, key_length]
        if attn_mask is not None:
            attn_mask = tf.cond(pred=tf.greater(num_beams, 1),
                                true_fn=lambda: tf.tile(attn_mask, [num_beams, 1, 1, 1]),
                                false_fn=lambda: attn_mask)

            ################################################## PRINTS ########################################################
            # print_ops = []
            # enc_dec = "decoder" if isDecoder else "encoder"
            # if self.name == "self_attn_sublayer" and not isDecoder:
            #     print_ops.append(
            #         tf.compat.v1.Print([], [tf.shape(attn_mask)],
            #                            "AVIVSL7: in " + enc_dec + " attn_mask shape " + self.name, summarize=10000))
            #     print_ops.append(
            #         tf.compat.v1.Print([], [attn_mask[:, 0, :, :]],
            #                            "AVIVSL7: in " + enc_dec + " attn_mask[:,0,:,:] " + self.name, summarize=10000))
            #     print_ops.append(
            #         tf.compat.v1.Print([], [attn_mask[:, 1, :, :]],
            #                            "AVIVSL7: in " + enc_dec + " attn_mask[:,1,:,:] " + self.name, summarize=10000))
            #     print_ops.append(
            #         tf.compat.v1.Print([], [attn_mask[:, 2, :, :]],
            #                            "AVIVSL7: in " + enc_dec + " attn_mask[:,2,:,:] " + self.name, summarize=10000))
            #     print_ops.append(
            #          tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[0, 0, :, :]],
            #                             "AVIVSL7: in " + enc_dec + " attn_logits (with shape) before mask" + self.name, summarize=10000))
            #     print_ops.append(
            #          tf.compat.v1.Print([], [tf.shape(values), values[0, 0, :, :10]], "values " + self.name, 50, 100))
            # # print_ops.append(
            # #     tf.compat.v1.Print([], [tf.shape(queries), queries[0, 0, :, :10]], "queries " + self.name, 50, 100))
            # # print_ops.append(tf.compat.v1.Print([], [tf.shape(keys), keys[0, 0, :, :10]], "keys " + self.name, 50, 100))
            # # if "cross" in self.name:
            # #     print_ops = []
            # with tf.control_dependencies(print_ops):
            #     attn_logits = attn_logits * 1
            #################################################################################################################

            attn_logits += attn_mask

            ################################################## PRINTS ########################################################
            # print_ops = []
            # if isDecoder and 'self_attn' in self.name :
            #     train_or_inference = 'inference' if isInference else 'train'
            #     print_ops.append(
            #                 tf.compat.v1.Print([], [tf.shape(attn_logits)],
            #                                    "AVIVSL7 attn_logits in " + self.name + ' in ' + train_or_inference, summarize=10000))
            # enc_dec = "decoder" if isDecoder else "encoder"
            # if self.name == "self_attn_sublayer" and not isDecoder:
            #     print_ops.append(
            #         tf.compat.v1.Print([], [tf.shape(attn_mask),attn_mask[:,0,:,:]],
            #                            "AVIVSL20: in " + enc_dec + " attn_mask" + self.name,
            #                            summarize=10000))
            # with tf.control_dependencies(print_ops):
            #     attn_logits = attn_logits * 1
            #################################################################################################################
            # to see the order of sentences in the batch, search for "PRINT TO SEE THE ORDER OF SENTENCES" in transformer.py



        #if pre_softmax_scaled_attn_mask is not None:

            ################################################## PRINTS ######################################################
            # print_ops = []
            # if pre_softmax_scaled_attn_mask is not None:
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask), pre_softmax_scaled_attn_mask[:,0,:,:]],
            #                                "AVIVSL7 pre_softmax_scaled_attn_mask[:,0,:,:]: ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask), pre_softmax_scaled_attn_mask[:,1,:,:]],
            #                                "AVIVSL7 pre_softmax_scaled_attn_mask[:,1,:,:]: ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask), pre_softmax_scaled_attn_mask[:,2,:,:]],
            #                                "AVIVSL7 pre_softmax_scaled_attn_mask[:,2,:,:]: ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask), pre_softmax_scaled_attn_mask[:,3,:,:]],
            #                                "AVIVSL7 pre_softmax_scaled_attn_mask[:,3,:,:]: ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask), pre_softmax_scaled_attn_mask[:,4,:,:]],
            #                                "AVIVSL7 pre_softmax_scaled_attn_mask[:,4,:,:]: ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask), pre_softmax_scaled_attn_mask[:,5,:,:]],
            #                                "AVIVSL7 pre_softmax_scaled_attn_mask[:,5,:,:]: ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask), pre_softmax_scaled_attn_mask[:,6,:,:]],
            #                                "AVIVSL7 pre_softmax_scaled_attn_mask[:,6,:,:]: ", summarize=10000))
                # print_ops.append(
                #         tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[:,0,:,:]],
                #                            "AVIVSL8 attn_logits[:,0,:,:] before: ", summarize=10000))
                # print_ops.append(
                #         tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[:,2,:,:]],
                #                            "AVIVSL8 attn_logits[:,2,:,:] before: ", summarize=10000))
                # print_ops.append(
                #         tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[:,3,:,:]],
                #                            "AVIVSL8 attn_logits[:,3,:,:] before: ", summarize=10000))
            # with tf.control_dependencies(print_ops):
            #     attn_logits = attn_logits * 1
            ################################################################################################################

            if pre_softmax_scaled_attn_mask is not None:
                pre_softmax_scaled_attn_mask = tf.dtypes.cast(pre_softmax_scaled_attn_mask, attn_logits.dtype)
                attn_logits *= pre_softmax_scaled_attn_mask

            ################################################## PRINTS ######################################################
            # print_ops = []
            # if pre_softmax_scaled_attn_mask is not None:
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[:,0,:,:]],
            #                                "AVIVSL9 attn_logits[:,0,:,:] after: ", summarize=10000))
            # print_ops.append(
            #         tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[:,2,:,:]],
            #                            "AVIVSL9 attn_logits[:,2,:,:] after: ", summarize=10000))
            # print_ops.append(
            #         tf.compat.v1.Print([], [tf.shape(attn_logits), attn_logits[:,3,:,:]],
            #                            "AVIVSL9 attn_logits[:,3,:,:] after: ", summarize=10000))
            # with tf.control_dependencies(print_ops):
            #     attn_logits = attn_logits * 1
            ################################################################################################################

            ################################################## PRINTS ######################################################
            # print_ops = []
            # if pre_softmax_scaled_attn_mask is not None:
            #     print_ops.append(
            #             tf.compat.v1.Print([], [tf.shape(pre_softmax_scaled_attn_mask)],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask shape: ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [pre_softmax_scaled_attn_mask[:,0,:,:]],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask[:,0,:,:] ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [pre_softmax_scaled_attn_mask[:,1,:,:]],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask[:,1,:,:] ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [pre_softmax_scaled_attn_mask[:,2,:,:]],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask[:,2,:,:] ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [pre_softmax_scaled_attn_mask[:,3,:,:]],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask[:,3,:,:] ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [pre_softmax_scaled_attn_mask[:,4,:,:]],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask[:,4,:,:] ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [pre_softmax_scaled_attn_mask[:,5,:,:]],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask[:,5,:,:] ", summarize=10000))
            #     print_ops.append(
            #             tf.compat.v1.Print([], [pre_softmax_scaled_attn_mask[:,6,:,:]],
            #                                "AVIVSL9 pre_softmax_scaled_attn_mask[:,6,:,:] ", summarize=10000))
            # with tf.control_dependencies(print_ops):
            #     attn_logits = attn_logits * 1
            ################################################################################################################


        # Calculate attention weights
        attn_weights = tf.nn.softmax(attn_logits)
        if post_softmax_scaled_attn_mask is not None:
            post_softmax_scaled_attn_mask = tf.dtypes.cast(post_softmax_scaled_attn_mask, attn_weights.dtype)
            attn_weights *= post_softmax_scaled_attn_mask

        if self.config.target_same_scene_head_FC_FFN and self.name=='self_attn_sublayer' and self.isDecoder and self.layer_id in self.target_same_scene_mask_projection_layers:
            target_same_scene_learnt_mask, num_of_heads = self._get_target_mask(target_mask_learning)
            target_same_scene_learnt_mask = target_same_scene_learnt_mask[:,:,:,:tf.shape(attn_weights)[3]] # cut the mask to be in the length of the current sentence
            target_same_scene_learnt_mask_all_heads = self._get_target_mask_all_heads(target_same_scene_learnt_mask, num_of_heads)
            attn_weights*=target_same_scene_learnt_mask_all_heads
        else:
            target_same_scene_learnt_mask = None

        # ############################################### PRINTING #######################################################
        # printops = []
        # inference_or_train = 'inference' if isInference else 'train'
        # if target_same_scene_learnt_mask is not None:
        #     if self.isDecoder and self.name=='self_attn_sublayer':
        #         printops.append(tf.compat.v1.Print([], [self.layer_id], "AVIVSL7: layer_id " + inference_or_train, summarize=10000))
        # if target_same_scene_learnt_mask is not None:
        #     printops.append(tf.compat.v1.Print([], [self.layer_id], "AVIVSL7: layer_id: ",summarize=10000))
            # printops.append(
            #     tf.compat.v1.Print([], [tf.shape(target_mask_learning[0])], "AVIVSL7: target_mask_learning_input ",
            #                        summarize=10000))
            # printops.append(tf.compat.v1.Print([], [target_mask_learning[1]], "AVIVSL7: target_mask_learning_num_heads ", summarize=10000))
            # printops.append(
            #     tf.compat.v1.Print([], [tf.shape(target_same_scene_learnt_mask)], "AVIVSL7: target_same_scene_learnt_mask ",
            #                        summarize=10000))
            # printops.append(tf.compat.v1.Print([], [tf.shape(target_mask[:,:,:,:tf.shape(attn_weights)[3]])],"AVIVSL7: target_mask[:,:,:,:tf.shape(attn_weights)[3]] ", summarize=10000))
        # with tf.control_dependencies(printops):
        #     attn_weights = attn_weights * 1
        ################################################################################################################

        undropped_attn_weights = attn_weights

        # Optionally apply dropout:
        if self.dropout_attn > 0.0:
            attn_weights = tf.compat.v1.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Weigh attention values
        weighted_memories = tf.matmul(attn_weights, values)
        # ############################################### PRINTING #######################################################
        # printops = []
        # if self.name == 'cross_attn_sublayer':
        #     printops.append(tf.compat.v1.Print([], [tf.shape(attn_weights), attn_weights[:,0,:,:]],
        #                                    "AVIVSL7: attn_weights ", summarize=10000))
        #     printops.append(tf.compat.v1.Print([], [tf.shape(values), values[:,0,:,:]],
        #                                    "AVIVSL7: values ", summarize=10000))
        #     printops.append(tf.compat.v1.Print([], [tf.shape(weighted_memories), weighted_memories[:,0,:,:]],
        #                                    "AVIVSL7: weighted_memories ", summarize=10000))
        # with tf.control_dependencies(printops):
        #     weighted_memories = weighted_memories * 1
        ################################################################################################################
        return weighted_memories, undropped_attn_weights, target_same_scene_learnt_mask

    def _subtitute_heads(self, head_set, substitutes):
        """
        substitues heads of keys/values/queries with pre-defined heads
        :param head_set: the full keys/values/queries head set, with shape: [#batches, sententce_len, #heads, embedding size/#heads]
        :param substitutes: list of tuples, each containing: (head to subtitute, #of those heads to subtitute)
        :return: the updated full keys/values/queries head set
        """
        total_head_subs = 0
        updated_head_set = None
        # first concatenating all the extra new heads
        for elem in substitutes:
            full_head=elem[0]
            num_of_heads = elem[1]
            full_head = tf.transpose(a=full_head, perm=[0, 2, 1, 3])
            total_head_subs+=num_of_heads
            # concatenting all the heads of the current head (each time taking the i_th piece of embedding size/#heads of it
            for i in list(range(num_of_heads)):
                head_embedding_size = tf.shape(head_set)[3]
                curr_head = full_head[:,:,:,(i*head_embedding_size):(((i+1)*head_embedding_size))]
                if updated_head_set is None:
                    updated_head_set = curr_head
                else:
                    updated_head_set = tf.concat([updated_head_set,curr_head], axis=2)

            ############################################### PRINTS #################################################################
            # print_ops = []
            # if updated_head_set is None:
            #     print_ops.append(
            #         tf.compat.v1.Print([], [], "AVIVSL10: it's None!", summarize=10000))
            # else:
            #     print_ops.append(
            #         tf.compat.v1.Print([], [tf.shape(updated_head_set)], "AVIVSL10: updated_head_set before", summarize=10000))
            # print_ops.append(
            #     tf.compat.v1.Print([], [tf.shape(head_set[:,:,total_head_subs:,:])], "AVIVSL10: head_set[:,:,total_head_subs:,:] ", summarize=10000))
            # with tf.control_dependencies(print_ops):
            #     head_set = head_set * 1
            #########################################################################################################################

            if updated_head_set is None:
                updated_head_set = head_set
            else: # then adding the part of the original head_set that wasn't substitued
                updated_head_set = tf.concat([updated_head_set, head_set[:,:,total_head_subs:,:]], axis=2)

            ############################################### PRINTS #################################################################
            # print_ops = []
            # print_ops.append(
            #     tf.compat.v1.Print([], [tf.shape(updated_head_set)], "AVIVSL10: updated_head_set after", summarize=10000))
            # with tf.control_dependencies(print_ops):
            #     updated_head_set = updated_head_set * 1
            #########################################################################################################################

            return updated_head_set

    def _get_target_mask(self, target_mask_learning):
        FFN_input=target_mask_learning[0]
        num_of_heads = target_mask_learning[1]
        target_mask = self.target_same_scene_mask_projection.forward(FFN_input)
        return target_mask, num_of_heads

    def _get_target_mask_all_heads(self, target_same_scene_learnt_mask, num_of_heads):
        """
        :target_same_scene_learnt_mask: the target learnt mask
        :num_of_heads: number of heads on which to impose the target_same_scene_learnt_mask
        :return: a full set of all head masks, where target_same_scene_learnt_mask is duplicated num_of_heads times, and the rest of the heads get a "mask" of ones (no mask)
        """
        masks_list = [target_same_scene_learnt_mask for _ in range(num_of_heads)] \
                    + [tf.ones_like(target_same_scene_learnt_mask) for _ in range(self.config.transformer_num_heads - num_of_heads)]
        res_attn_mask = tf.concat(masks_list, axis=1)
        return res_attn_mask



    def forward(self, query_context, memory_context, attn_mask, pre_softmax_scaled_attn_mask,post_softmax_scaled_attn_mask, layer_memories, cross_attention_keys_update_rules, target_mask_learning, isDecoder=False, isInference=False):
        """ Propagates the input information through the attention layer. """
        # The context for the query and the referenced memory is identical in case of self-attention
        if memory_context is None:
            memory_context = query_context
        ############################################### PRINTS #################################################################
        # print_ops = []
        # if isDecoder:
        #     if isInference:
        #         print_ops.append(
        #                 tf.compat.v1.Print([], [], "Inference!", 50, 100))
        #         print_ops.append(
        #                 tf.compat.v1.Print([], [tf.shape(memory_context), tf.shape(query_context)], "memory_context and query_context in inference", 50, 100))
        #     else:
        #         print_ops.append(
        #                 tf.compat.v1.Print([], [tf.shape(memory_context), tf.shape(query_context)], "memory_context and query_context in train", 50, 100))
        # print_ops.append(
        #     tf.compat.v1.Print([], [tf.shape(memory_context), memory_context[..., :10]], "in_memory_context(key+val)" + self.name, 50, 100))
        # print_ops.append(
        #     tf.compat.v1.Print([], [tf.shape(memory_context), tf.shape(query_context)], "same?" + self.name, 50, 100))
        # if isDecoder:
        #     print_ops.append(
        #          tf.compat.v1.Print([], [tf.shape(query_context), query_context[..., :10]], "AVIVSL6: in_query_context " + self.name, 50, 100))
        #     print_ops.append(
        #          tf.compat.v1.Print([], [tf.shape(attn_mask), attn_mask], "AVIVSL7: attn_mask " + self.name, 50, 100))
        # if "self" in self.name:
        #     print_ops = []
        # with tf.control_dependencies(print_ops):
        #     query_context = query_context * 1
        #########################################################################################################################
        # Get attention inputs
        queries, keys, values = self._compute_attn_inputs(query_context, memory_context) ## size: values+keys: [3 19 256] queries: [3 24 256] source language sent len:19, target language sent len:24

        ############################################### PRINTS #################################################################
        # print_ops = []
        # if isDecoder and isInference and self.name=='self_attn_sublayer':
        #     inference_or_train = 'inference' if isInference else 'train'
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(keys)], "AVIVSL6:keys " + self.name + ' in ' + inference_or_train,summarize=10000))
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(queries)], "AVIVSL6:queries " + self.name + ' in ' + inference_or_train,summarize=10000))
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(values)], "AVIVSL6:values " + self.name + ' in ' + inference_or_train,summarize=10000))
        # with tf.control_dependencies(print_ops):
        #     queries = queries * 1
        #########################################################################################################################


        # Recall and update memories (analogous to the RNN state) - decoder only
        if layer_memories is not None:
            keys = tf.concat([layer_memories['keys'], keys], axis=1)
            values = tf.concat([layer_memories['values'], values], axis=1)
            layer_memories['keys'] = keys
            layer_memories['values'] = values

        # Split attention inputs among attention heads
        split_queries = self._split_among_heads(queries) ## size: queries: [3 24 8 32] target language sent len:24
        split_keys = self._split_among_heads(keys) ## size: [3 19 8 32] source language sent len:19
        split_values = self._split_among_heads(values) ## size: [3 19 8 32] source language sent len:19

        if cross_attention_keys_update_rules is not None:
            split_keys = self._subtitute_heads(split_keys, cross_attention_keys_update_rules)

        ############################################### PRINTS #################################################################
        # print_ops = []
        # inference_or_train = 'inference' if isInference else 'train'
        # if isInference and self.name=='self_attn_sublayer':
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(split_values)], "AVIVSL20: split_values in " + self.name, summarize=10000))
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(split_queries)], "AVIVSL20: split_queries in " + self.name, summarize=10000))

        # if attn_mask is not None and self.name=='cross_attn_sublayer':
            # print_ops.append(
            #      tf.compat.v1.Print([], [tf.shape(split_queries)], "AVIVSL20: split_queries in " + self.name, summarize=10000))
            # print_ops.append(
            #      tf.compat.v1.Print([], [tf.shape(split_keys[:,:,:,tf.shape(split_keys)[3]:tf.shape(split_keys)[3]+3])], "AVIVSL20: split_keys in " + self.name, summarize=10000))
        #     print_ops.append(
        #          tf.compat.v1.Print([], [tf.shape(split_values)], "AVIVSL20: split_values in " + self.name, summarize=10000))
        # # #     #     print_ops.append(
        # # #     #          tf.compat.v1.Print([], [tf.shape(attn_mask), attn_mask], "AVIVSL7: attn_mask " + self.name, 50, 100))
        # # #     # if "self" in self.name:
        # # #     #     print_ops = []
        # with tf.control_dependencies(print_ops):
        #     split_queries = split_queries * 1
        ########################################################################################################################


        # Apply attention function
        split_weighted_memories, attn_softmax_weights, target_same_scene_learnt_mask = self._dot_product_attn(split_queries, split_keys, split_values, attn_mask, pre_softmax_scaled_attn_mask,post_softmax_scaled_attn_mask,target_mask_learning,
                                                         scaling_on=True, isDecoder=isDecoder, isInference=isInference)
        # Merge head output
        weighted_memories = self._merge_from_heads(split_weighted_memories)
        # Feed through a dense layer
        projected_memories = self.context_projection.forward(weighted_memories)
        ############################################### PRINTS #################################################################
        print_ops = []
        # if isDecoder and self.name=="self_attn_sublayer":
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(weighted_memories), weighted_memories],
        #                                         "AVIVSL7: decoder self_attn weighted_memories:" + self.name,summarize=10000))
        # if isDecoder and self.name=="self_attn_sublayer":
        #     print_ops.append(tf.compat.v1.Print([], [tf.shape(attn_mask), attn_mask],
        #                                         "AVIVSL8: decoder attn_mask:" + self.name,summarize=10000))
        # with tf.control_dependencies(print_ops):
        #     projected_memories = projected_memories * 1
        #########################################################################################################################

        return projected_memories, layer_memories, attn_softmax_weights, target_same_scene_learnt_mask


class SingleHeadAttentionLayer(object):
    """ Single-head attention module. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 hidden_dims,
                 float_dtype,
                 dropout_attn,
                 training,
                 name,
                 attn_type='multiplicative'):

        # Declare attributes
        self.reference_dims = reference_dims
        self.hypothesis_dims = hypothesis_dims
        self.hidden_dims = hidden_dims
        self.float_dtype = float_dtype
        self.dropout_attn = dropout_attn
        self.attn_type = attn_type
        self.training = training
        self.name = name

        assert attn_type in ['additive', 'multiplicative'], 'Attention type {:s} is not supported.'.format(attn_type)

        # Instantiate parameters
        with tf.compat.v1.variable_scope(self.name):
            self.queries_projection = None
            self.attn_weight = None
            if attn_type == 'additive':
                self.queries_projection = FeedForwardLayer(self.hypothesis_dims,
                                                           self.hidden_dims,
                                                           float_dtype,
                                                           dropout_rate=0.,
                                                           activation=None,
                                                           use_bias=False,
                                                           use_layer_norm=False,
                                                           training=self.training,
                                                           name='queries_projection')

                self.attn_weight = tf.compat.v1.get_variable(name='attention_weight',
                                                             shape=self.hidden_dims,
                                                             dtype=float_dtype,
                                                             initializer=glorot_uniform_initializer(),
                                                             trainable=True)

            self.keys_projection = FeedForwardLayer(self.reference_dims,
                                                    self.hidden_dims,
                                                    float_dtype,
                                                    dropout_rate=0.,
                                                    activation=None,
                                                    use_bias=False,
                                                    use_layer_norm=False,
                                                    training=self.training,
                                                    name='keys_projection')

    def _compute_attn_inputs(self, query_context, memory_context):
        """ Computes query, key, and value tensors used by the attention function for the calculation of the
        time-dependent context representation. """
        queries = query_context
        if self.attn_type == 'additive':
            queries = self.queries_projection.forward(query_context)
        keys = self.keys_projection.forward(memory_context)
        values = memory_context
        return queries, keys, values

    def _additive_attn(self, queries, keys, values, attn_mask):
        """ Uses additive attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes time-step-wise attention scores. """
            query = tf.expand_dims(query, 1)
            return tf.reduce_sum(input_tensor=self.attn_weight * tf.nn.tanh(keys + query), axis=-1)

        # Obtain attention scores
        transposed_queries = tf.transpose(a=queries, perm=[1, 0, 2])  # time-major
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)
        attn_logits = tf.transpose(a=attn_logits, perm=[1, 0, 2])

        if attn_mask is not None:
            # Transpose and tile the mask
            attn_logits += tf.tile(tf.squeeze(attn_mask, 1),
                                   [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1])

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-1, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.compat.v1.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Obtain context vectors
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def _multiplicative_attn(self, queries, keys, values, attn_mask):
        """ Uses multiplicative attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        # Use multiplicative attention
        transposed_keys = tf.transpose(a=keys, perm=[0, 2, 1])
        attn_logits = tf.matmul(queries, transposed_keys)
        if attn_mask is not None:
            # Transpose and tile the mask
            attn_logits += tf.tile(tf.squeeze(attn_mask, 1),
                                   [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1])

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-1, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.compat.v1.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Obtain context vectors
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def forward(self, query_context, memory_context, attn_mask, layer_memories):
        """ Propagates the input information through the attention layer. """
        # The context for the query and the referenced memory is identical in case of self-attention
        if memory_context is None:
            memory_context = query_context

        # Get attention inputs
        queries, keys, values = self._compute_attn_inputs(query_context, memory_context)
        # Recall and update memories (analogous to the RNN state) - decoder only
        if layer_memories is not None:
            keys = tf.concat([layer_memories['keys'], keys], axis=1)
            values = tf.concat([layer_memories['values'], values], axis=1)
            layer_memories['keys'] = keys
            layer_memories['values'] = values

        # Obtain weighted layer hidden representations
        if self.attn_type == 'additive':
            weighted_memories = self._additive_attn(queries, keys, values, attn_mask)
        else:
            weighted_memories = self._multiplicative_attn(queries, keys, values, attn_mask)
        return weighted_memories, layer_memories


class FineGrainedAttentionLayer(SingleHeadAttentionLayer):
    """ Defines the fine-grained attention layer; based on
    "Fine-grained attention mechanism for neural machine translation.", Choi et al, 2018. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 hidden_dims,
                 float_dtype,
                 dropout_attn,
                 training,
                 name,
                 attn_type='multiplicative'):
        super(FineGrainedAttentionLayer, self).__init__(reference_dims, hypothesis_dims, hidden_dims, float_dtype,
                                                        dropout_attn, training, name, attn_type)

    def _additive_attn(self, queries, keys, values, attn_mask):
        """ Uses additive attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes time-step-wise attention scores. """
            query = tf.expand_dims(query, 1)
            return self.attn_weight * tf.nn.tanh(keys + query)

        # Obtain attention scores
        transposed_queries = tf.transpose(a=queries, perm=[1, 0, 2])  # time-major
        # attn_logits has shape=[time_steps_q, batch_size, time_steps_k, num_features]
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)

        if attn_mask is not None:
            transposed_mask = \
                tf.transpose(
                    a=tf.tile(attn_mask, [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1, 1]),
                    perm=[2, 0, 3, 1])
            attn_logits += transposed_mask

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-2, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.compat.v1.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)

        # Obtain context vectors
        expanded_values = tf.expand_dims(values, axis=1)
        weighted_memories = \
            tf.reduce_sum(input_tensor=tf.multiply(tf.transpose(a=attn_weights, perm=[1, 0, 2, 3]), expanded_values),
                          axis=2)
        return weighted_memories

    def _multiplicative_attn(self, queries, keys, values, attn_mask):
        """ Uses multiplicative attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes time-step-wise attention scores. """
            query = tf.expand_dims(query, 1)
            return tf.multiply(keys, query)

        # Obtain attention scores
        transposed_queries = tf.transpose(a=queries, perm=[1, 0, 2])  # time-major
        # attn_logits has shape=[time_steps_q, batch_size, time_steps_k, num_features]
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)

        if attn_mask is not None:
            transposed_mask = \
                tf.transpose(
                    a=tf.tile(attn_mask, [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1, 1]),
                    perm=[2, 0, 3, 1])
            attn_logits += transposed_mask

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-2, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.compat.v1.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)

        # Obtain context vectors
        expanded_values = tf.expand_dims(values, axis=1)
        weighted_memories = \
            tf.reduce_sum(input_tensor=tf.multiply(tf.transpose(a=attn_weights, perm=[1, 0, 2, 3]), expanded_values),
                          axis=2)
        return weighted_memories

    def _attn(self, queries, keys, values, attn_mask):
        """ For each encoder layer, weighs and combines time-step-wise hidden representation into a single layer
        context state.  -- DEPRECATED, SINCE IT'S SLOW AND PROBABLY NOT ENTIRELY CORRECT """
        # Account for beam-search
        num_beams = tf.shape(input=queries)[0] // tf.shape(input=keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes position-wise attention scores. """
            query = tf.expand_dims(query, 1)
            # return tf.squeeze(self.attn_weight * (tf.nn.tanh(keys + query + norm_bias)), axis=2)
            return self.attn_weight * tf.nn.tanh(keys + query)  # 4D output

        def _weighting_fn(step_weights):
            """ Computes position-wise context vectors. """
            # step_weights = tf.expand_dims(step_weights, 2)
            return tf.reduce_sum(input_tensor=tf.multiply(step_weights, values), axis=1)

        # Obtain attention scores
        transposed_queries = tf.transpose(a=queries, perm=[1, 0, 2])
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)  # multiple queries per step are possible
        if attn_mask is not None:
            # attn_logits has shape=[batch, query_lengh, key_length, attn_features]
            transposed_mask = \
                tf.transpose(
                    a=tf.tile(attn_mask, [tf.shape(input=queries)[0] // tf.shape(input=attn_mask)[0], 1, 1, 1]),
                    perm=[2, 0, 3, 1])
            attn_logits += transposed_mask

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-2, name='attn_weights')

        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.compat.v1.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)

        # Obtain context vectors
        weighted_memories = tf.map_fn(_weighting_fn, attn_weights)
        weighted_memories = tf.transpose(a=weighted_memories, perm=[1, 0, 2])
        return weighted_memories
