"""Utility functions."""

import pickle as pkl
import json
import logging
import numpy
import sys
import numpy as np
import tensorflow as tf

try:
    from . import exception
except (ModuleNotFoundError, ImportError) as e:
    import exception


def reset_dict_vals(d):
    """
    :param d: dictionary
    :return: vals are consecutive numbers starting from 0, sorted by the order of the original vals
    """
    return {key: i for i, (key, val) in enumerate(sorted(d.items(), key=lambda x:x[1]))}


def reset_dict_indexes(d):
    """
    :param d: dictionary
    :return: keys are consecutive numbers starting from 0, sorted by the order of the keys
    """
    return {i: d[key] for i, key in enumerate(sorted(d.keys()))}
# batch preparation
# def prepare_data(seqs_x, seqs_y, seq_edges, seq_labels, seq_edges_time, seq_labels_time, n_factors, maxlen=None):
def prepare_data(seqs_x, seqs_y, seq_edges_time, seq_labels_time, n_factors, maxlen=None):

    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    # move edges to length major instead of batch major
    if seq_edges_time is not None:
        # target_edges = numpy.array(seq_edges)
        # target_edges = numpy.moveaxis(target_edges, 0, -1)
        # target_labels = numpy.array(seq_labels)
        # target_labels = numpy.moveaxis(target_labels, 0, -1)
        target_edges_time = numpy.array(seq_edges_time)
        target_edges_time = numpy.moveaxis(target_edges_time, 0, -1)
        target_labels_time = numpy.array(seq_labels_time)
        target_labels_time = numpy.moveaxis(target_labels_time, 0, -1)
    else:
        # target_edges = None
        # target_labels = None
        target_edges_time = None
        target_labels_time = None

    # drop pairs with sentences longer than maxlen
    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        kept = []
        for i, (l_x, s_x, l_y, s_y) in enumerate(zip(lengths_x, seqs_x, lengths_y, seqs_y)):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                kept.append(i)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        if seq_edges_time is not None:
            # target_edges = target_edges[:,:,:,kept]
            # target_labels = target_labels[:,:,:,kept]
            target_edges_time = target_edges_time[:,:,:,kept]
            target_labels_time = target_labels_time[:,:,:,kept]
        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1
    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = list(zip(*s_x))
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask, target_edges_time, target_labels_time
    # return x, x_mask, y, y_mask, target_edges, target_labels, target_edges_time, target_labels_time


def array_to_sparse_tensor(ar, base_val=float("inf")):
    # indices = np.nonzero(ar != base_val)
    # values = ar[indices]
    # shape = ar.shape
    # indices = np.transpose(indices)
    # sparse = tf.compat.v1.SparseTensorValue(indices=indices, values=values, dense_shape=shape) #TODO why are the inf in the values?
    sparse = dense_to_sparse_tensor(tf.convert_to_tensor(ar), base_val=base_val)
    return sparse


def dense_to_sparse_tensor(dns, base_val=float("inf")):
    """
    convert a dense tensor to a sparse one
    :param dns: a tensor (currently dynamic shape is not supported)
    :param base_val: which values to remove from the tensor (default float("inf")
    :return:
    """
    # Find indices where the tensor is not zero
    idx = tf.where(tf.not_equal(dns, base_val))

    # Make the sparse tensor
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape()
    # if tensor shape is dynamic
    printops = [tf.Print([], [idx, tf.gather_nd(dns, idx)], "sanity", 10, 50)]
    with tf.control_dependencies(printops):
        # sparse = tf.compat.v1.SparseTensor(indices=idx, values=tf.gather_nd(dns, idx), dense_shape=dns.get_shape())
        sparse = tf.compat.v1.SparseTensor(indices=idx, values=tf.gather_nd(dns, idx), dense_shape=tf.cast(tf.shape(dns), tf.int64))
    return sparse


def load_dict(filename, model_type):
    try:
        # build_dictionary.py writes JSON files as UTF-8 so assume that here.
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except:
        # FIXME Should we be assuming UTF-8?
        with open(filename, 'r', encoding='utf-8') as f:
            d = pkl.load(f)

    # The transformer model requires vocab dictionaries to use the new style
    # special symbols. If the dictionary looks like an old one then tell the
    # user to update it.
    if model_type == 'transformer' and ("<GO>" not in d or d["<GO>"] != 1):
        logging.error('you must update \'{}\' for use with the '
                      '\'transformer\' model type. Please re-run '
                      'build_dictionary.py to generate a new vocabulary '
                      'dictionary.'.format(filename))
        sys.exit(1)

    return d


def seq2words(seq, inverse_dictionary, join=True):
    seq = numpy.array(seq, dtype='int64')
    assert len(seq.shape) == 1
    return factoredseq2words(seq.reshape([seq.shape[0], 1]),
                             [inverse_dictionary],
                             join)


def factoredseq2words(seq, inverse_dictionaries, join=True):
    assert len(seq.shape) == 2
    assert len(inverse_dictionaries) == seq.shape[1]
    words = []
    eos_reached = False
    for i, w in enumerate(seq):
        if eos_reached:
            break
        factors = []
        for j, f in enumerate(w):
            if f == 0:
                eos_reached = True
                break
                # This assert has been commented out because it's possible for
                # non-zero values to follow zero values for Transformer models.
                # TODO Check why this happens
                #assert (i == len(seq) - 1) or (seq[i+1][j] == 0), \
                #       ('Zero not at the end of sequence', seq)
            elif f in inverse_dictionaries[j]:
                factors.append(inverse_dictionaries[j][f])
            else:
                factors.append('UNK')
        word = '|'.join(factors)
        words.append(word)
    return ' '.join(words) if join else words


def reverse_dict(dictt):
    keys, values = list(zip(*list(dictt.items())))
    r_dictt = dict(list(zip(values, keys)))
    return r_dictt


def load_dictionaries(config):
    model_type = config.model_type
    source_to_num = [load_dict(d, model_type) for d in config.source_dicts]
    target_to_num = load_dict(config.target_dict, model_type)
    num_to_source = [reverse_dict(d) for d in source_to_num]
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target


def read_all_lines(config, sentences, batch_size):
    source_to_num, _, _, _ = load_dictionaries(config)

    if config.source_vocab_sizes != None:
        assert len(config.source_vocab_sizes) == len(source_to_num)
        for d, vocab_size in zip(source_to_num, config.source_vocab_sizes):
            if vocab_size != None and vocab_size > 0:
                for key, idx in list(d.items()):
                    if idx >= vocab_size:
                        del d[key]

    lines = []
    for sent in sentences:
        line = []
        for w in sent.strip().split():
            if config.factors == 1:
                w = [source_to_num[0][w] if w in source_to_num[0] else 2]
            else:
                w = [source_to_num[i][f] if f in source_to_num[i] else 2
                                         for (i,f) in enumerate(w.split('|'))]
                if len(w) != config.factors:
                    raise exception.Error(
                        'Expected {0} factors, but input word has {1}\n'.format(
                            config.factors, len(w)))
            line.append(w)
        lines.append(line)
    lines = numpy.array(lines)
    lengths = numpy.array([len(l) for l in lines])
    idxs = lengths.argsort()
    lines = lines[idxs]

    #merge into batches
    batches = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        batches.append(batch)

    return batches, idxs
