import logging
import sys
import time

import numpy

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import exception
    from . import util
    from .parsing.corpus import extract_text_from_combined_tokens
except (ModuleNotFoundError, ImportError) as e:
    import exception
    import util
    from parsing.corpus import extract_text_from_combined_tokens


def translate_batch(session, sampler, x, x_mask, max_translation_len,
                    normalization_alpha, same_scene_mask, parent_scaled_mask=None): #TODO: AVIVSL make sure everyone calling it passes a parent_scaled mask
    """Translate a batch using a RandomSampler or BeamSearchSampler.

    Args:
        session: a TensorFlow session.
        sampler: a BeamSearchSampler or RandomSampler object.
        x: input Tensor with shape (factors, max_seq_len, batch_size).
        x_mask: mask Tensor for x with shape (max_seq_len, batch_size).
        max_translation_len: integer specifying maximum translation length.
        normalization_alpha: float specifying alpha parameter for length
            normalization.

    Returns:
        A list of lists of (translation, score) pairs. The outer list contains
        one list for each input sentence in the batch. The inner lists contain
        k elements (where k is the beam size), sorted by score in best-first
        order.
    """

    x_tiled = numpy.tile(x, reps=[1, 1, sampler.beam_size])
    x_mask_tiled = numpy.tile(x_mask, reps=[1, sampler.beam_size])
    #same_scene_mask_tiled = numpy.tile(same_scene_mask, reps=[1, 1, sampler.beam_size])

    feed_dict = {}
    # Feed inputs to the models.
    for model, config in zip(sampler.models, sampler.configs):
        if config.model_type == 'rnn':
            feed_dict[model.inputs.x] = x_tiled
            feed_dict[model.inputs.x_mask] = x_mask_tiled
        else:
            assert config.model_type == 'transformer'
            # Inputs don't need to be tiled in the Transformer because it
            # checks for different batch sizes in the encoder and decoder and
            # does its own tiling internally at the connection points.
            feed_dict[model.inputs.x] = x
            feed_dict[model.inputs.x_mask] = x_mask
            if same_scene_mask is not None:
                feed_dict[model.inputs.x_source_same_scene_mask] = same_scene_mask

            if parent_scaled_mask is not None:
                feed_dict[model.inputs.x_source_parent_scaled_mask] = parent_scaled_mask

            # logging.info(f"x, x_mask in translate {x.shape}, {x_mask.shape}, {x}, {x_mask}")
        feed_dict[model.inputs.training] = False

    # Feed inputs to the sampler.
    feed_dict[sampler.inputs.batch_size_x] = x.shape[-1]
    feed_dict[sampler.inputs.max_translation_len] = max_translation_len
    feed_dict[sampler.inputs.normalization_alpha] = normalization_alpha

    # Run the sampler.
    translations, scores = session.run(sampler.outputs, feed_dict=feed_dict)

    assert len(translations) == x.shape[-1]
    assert len(scores) == x.shape[-1]

    # Sort the translations by score. The scores are (optionally normalized)
    # log probs so higher values are better.
    beams = []
    for i in range(len(translations)):
        pairs = zip(translations[i], scores[i])
        beams.append(sorted(pairs, key=lambda pair: pair[1], reverse=True))

    return beams


def translate_file(input_file, output_file, same_scene_masks_file, parent_scaled_masks_file, session, sampler, config,
                   max_translation_len, normalization_alpha, nbest=False,
                   minibatch_size=80, maxibatch_size=20):
    """Translates a source file using a RandomSampler or BeamSearchSampler.

    Args:
        input_file: file object from which source sentences will be read.
        output_file: file object to which translations will be written.
        session: TensorFlow session.
        sampler: BeamSearchSampler or RandomSampler object.
        config: model config.
        max_translation_len: integer specifying maximum translation length.
        normalization_alpha: float specifying alpha parameter for length
            normalization.
        nbest: if True, produce n-best output with scores; otherwise 1-best.
        minibatch_size: minibatch size in sentences.
        maxibatch_size: number of minibatches to read and sort, pre-translation.
    """

    def translate_maxibatch(maxibatch, num_to_target, num_prev_translated, same_scene_batch, parent_scaled_batch):
        """Translates an individual maxibatch.

        Args:
            maxibatch: a list of sentences.
            num_to_target: dictionary mapping target vocabulary IDs to strings.
            num_prev_translated: the number of previously translated sentences.
        """

        # Sort the maxibatch by length and split into minibatches.
        try:
            minibatches, idxs = util.read_all_lines(config, maxibatch,
                                                    minibatch_size, same_scene_batch, parent_scaled_batch)
        except exception.Error as x:
            logging.error(x.msg)
            sys.exit(1)

        # Translate the minibatches and store the resulting beam (i.e.
        # translations and scores) for each sentence.
        beams = []
        for x in minibatches:
            x, same_scene_mask, parent_scaled_mask = zip(x)
            x = x[0]
            same_scene_mask = same_scene_mask[0] if same_scene_batch is not None else None
            parent_scaled_mask = parent_scaled_mask[0] if parent_scaled_batch is not None else None

            ######################### delete? ##########################
            # if same_scene_batch is not None:
            #     x, same_scene_mask = zip(x)
            #     x, same_scene_mask = x[0], same_scene_mask[0]
            # else:
            #     same_scene_mask = None
            ############################################################



            y_dummy = numpy.zeros(shape=(len(x), 1))
            x, x_mask, _, _, _, _, _,same_scene_mask, parent_scaled_mask, _ = util.prepare_data(x, y_dummy, None, None, None, config.factors,
                                                         source_same_scene_masks=same_scene_mask, source_parent_scaled_masks=parent_scaled_mask, target_same_scene_masks=None, maxlen=None) #FIXME: AVIVSL add source_parent_scaled_masks and send on parent_scaled_mask (what prepare_data returns), same as "same_scene_mask"
            # logging.info(f"Batch size {x.shape}, minibatch_size {minibatch_size}, maxibatch_size {maxibatch_size}")
            sample = translate_batch(session, sampler, x, x_mask,
                                     max_translation_len, normalization_alpha, same_scene_mask, parent_scaled_mask)
            beams.extend(sample)
            num_translated = num_prev_translated + len(beams)
            logging.info('Translated {} sents'.format(num_translated))

        # Put beams into the same order as the input maxibatch.
        tmp = numpy.array(beams, dtype=numpy.object)
        ordered_beams = tmp[idxs.argsort()]

        # Write the translations to the output file.
        for i, beam in enumerate(ordered_beams):
            if nbest:
                num = num_prev_translated + i
                for sent, cost in beam:
                    translation = util.seq2words(sent, num_to_target)
                    logging.info(f"Translation {translation}")
                    if config.valid_remove_parse:
                        translation = extract_text_from_combined_tokens(translation)
                    logging.info(f"Translation without {translation}")
                    line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                    output_file.write(line)
            else:
                best_hypo, cost = beam[0]
                line = util.seq2words(best_hypo, num_to_target) + '\n'
                # logging.info(line[:-1])
                output_file.write(line)
            if (i + 1) % 100:
                output_file.flush()

    _, _, _, num_to_target = util.load_dictionaries(config)

    logging.info("NOTE: Length of translations is capped to {}".format(
        max_translation_len))

    start_time = time.time()

    num_translated = 0
    maxibatch = []
    same_scene_batch = [] if same_scene_masks_file is not None else None
    parent_scaled_batch = [] if parent_scaled_masks_file is not None else None
    while True:
        line = input_file.readline()
        if same_scene_masks_file is not None:
            same_scene_mask_line = same_scene_masks_file.readline()
        else:
            same_scene_mask_line = None

        if parent_scaled_masks_file is not None:
            parent_scaled_mask_line = parent_scaled_masks_file.readline()
        else:
            parent_scaled_mask_line = None

        if line == "":
            if len(maxibatch) > 0:
                translate_maxibatch(maxibatch, num_to_target, num_translated,same_scene_batch, parent_scaled_batch)
                num_translated += len(maxibatch)
            break
        maxibatch.append(line)
        if same_scene_masks_file is not None:
            same_scene_batch.append(same_scene_mask_line)
        if parent_scaled_masks_file is not None:
            parent_scaled_batch.append(parent_scaled_mask_line)
        if len(maxibatch) == (maxibatch_size * minibatch_size):
            translate_maxibatch(maxibatch, num_to_target, num_translated,same_scene_batch, parent_scaled_batch)
            num_translated += len(maxibatch)
            maxibatch = []
            same_scene_batch = [] if same_scene_masks_file is not None else None
            parent_scaled_batch = [] if parent_scaled_masks_file is not None else None

    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(
        num_translated, duration, num_translated / duration))
