import numpy

import gzip

import shuffle
from util import load_dict
from parsing.corpus import extract_text_from_combined_tokens
from parsing.corpus import ConllSent
from util import reset_dict_vals
import numpy as np

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class FileWrapper(object):

    def __init__(self, fname):
        self.pos = 0
        self.lines = fopen(fname).readlines()
        self.lines = numpy.array(self.lines, dtype=numpy.object)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.lines):
            raise StopIteration
        l = self.lines[self.pos]
        self.pos += 1
        return l

    def reset(self):
        self.pos = 0

    def seek(self, pos):
        assert pos == 0
        self.pos = 0

    def readline(self):
        return next(self)

    def shuffle_lines(self, perm):
        self.lines = self.lines[perm]
        self.pos = 0

    def __len__(self):
        return len(self.lines)


class TextIterator:
    """Simple Bitext iterator."""

    def __init__(self, source, target,
                 source_dicts, target_dict,
                 model_type,
                 batch_size=128,
                 maxlen=100,
                 source_vocab_sizes=None,
                 target_vocab_size=None,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 token_batch_size=0,
                 keep_data_in_memory=False,
                 remove_parse=False,
                 target_graph=False,
                 target_labels_num=None):
        if keep_data_in_memory:
            self.source, self.target = FileWrapper(source), FileWrapper(target)
            if shuffle_each_epoch:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
        elif shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main(
                [self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict, model_type))
        self.target_dict = load_dict(target_dict, model_type)
        self.target_graph = target_graph
        self.target_labels_num = target_labels_num
        if self.target_graph:
            # TODO clean code when ready
            self.target_actions = {key: val
                                   for key, val in self.target_dict.items() if "@@|" in key}
            self.target_actions = reset_dict_vals(self.target_actions)
            #
            # print("found actions:", self.target_actions)
            self.target_labels = {key: val
                                  for key, val in self.target_dict.items() if "|@@" in key}
            self.target_labels = reset_dict_vals(self.target_labels)
            # print("found labels:", self.target_labels)
            # self.target_dict = {key: val for key, val in self.target_dict.items() if not (
            #     key in self.target_actions or key in self.target_labels)} # TODO is it the right thing to delete it from the target dict?
            # # self.target_dict = reset_dict_vals(self.target_dict)
        else:
            self.target_labels = None
            self.target_actions = None
        # Determine the UNK value for each dictionary (the value depends on
        # which version of build_dictionary.py was used).

        def determine_unk_val(d):
            if '<UNK>' in d and d['<UNK>'] == 2:
                return 2
            return 1

        self.source_unk_vals = [determine_unk_val(d)
                                for d in self.source_dicts]
        self.target_unk_val = determine_unk_val(self.target_dict)

        self.target_graph = target_graph
        self.remove_parse = remove_parse
        self.keep_data_in_memory = keep_data_in_memory
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.source_vocab_sizes = source_vocab_sizes
        self.target_vocab_size = target_vocab_size

        self.token_batch_size = token_batch_size

        if self.source_vocab_sizes != None:
            assert len(self.source_vocab_sizes) == len(self.source_dicts)
            for d, vocab_size in zip(self.source_dicts, self.source_vocab_sizes):
                if vocab_size != None and vocab_size > 0:
                    for key, idx in list(d.items()):
                        if idx >= vocab_size:
                            del d[key]

        if self.target_vocab_size != None and self.target_vocab_size > 0:
            for key, idx in list(self.target_dict.items()):
                if idx >= self.target_vocab_size:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            if self.keep_data_in_memory:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
            else:
                self.source, self.target = shuffle.main(
                    [self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        longest_source = 0
        longest_target = 0

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(
            self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()

                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                if self.remove_parse:
                    ss = extract_text_from_combined_tokens(ss)
                    tt = extract_text_from_combined_tokens(tt)
                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by source/target buffer length
            if self.sort_by_length:
                tlen = numpy.array([max(len(s), len(t)) for (
                    s, t) in zip(self.source_buffer, self.target_buffer)])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        def lookup_token(t, d, unk_val):
            return d[t] if t in d else unk_val

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [lookup_token(f, self.source_dicts[i],
                                          self.source_unk_vals[i])
                             for (i, f) in enumerate(w.split('|'))]
                    else:
                        w = [lookup_token(w, self.source_dicts[0],
                                          self.source_unk_vals[0])]
                    tmp.append(w)
                ss_indices = tmp
                source.append(ss_indices)

                # read from source file and map to word index
                tt = self.target_buffer.pop()

                if self.target_graph:
                    tt_edge_time, tt_label_time = convert_text_to_graph(
                        tt, self.maxlen, self.target_labels, self.target_labels_num)

                    tt_text = extract_text_from_combined_tokens(tt)
                    tt_indices = [lookup_token(w, self.target_dict,
                                               self.target_unk_val) for w in tt_text]
                    target.append((tt_indices, tt_edge_time, tt_label_time))

                else:
                    tt_indices = [lookup_token(w, self.target_dict,
                                               self.target_unk_val) for w in tt]
                    if self.target_vocab_size != None:
                        tt_indices = [w if w < self.target_vocab_size
                                      else self.target_unk_val
                                      for w in tt_indices]
                    target.append(tt_indices)

                longest_source = max(longest_source, len(ss_indices))
                longest_target = max(longest_target, len(tt_indices))

                if self.token_batch_size:
                    if len(source) * longest_source > self.token_batch_size or \
                            len(target) * longest_target > self.token_batch_size:
                        # remove last sentence pair (that made batch over-long)
                        source.pop()
                        target.pop()
                        self.source_buffer.append(ss)
                        self.target_buffer.append(tt)

                        break

                else:
                    if len(source) >= self.batch_size or \
                            len(target) >= self.batch_size:
                        break
        except IOError:
            self.end_of_data = True

        return source, target


def _last_word(idxs, tokens,
               graceful=False):  # TODO use "graceful" at inference time, deal with cases where there is no last word
    if not idxs:
        if not graceful:
            raise IndexError("Asked for last word on an empty buffer")
        return [], []
    i = -1
    while -i < len(idxs):
        i -= 1
        if not tokens[i].endswith("@@"):
            i += 1
            break
    return idxs[i:], tokens[i:]


def convert_text_to_graph(x_target, max_len, labels_dict, num_labels):
    # TODO connect reduce and labels to the popped out
    # TODO convert to work with indexes
    slf = 0
    lft = 1
    right = 2
    # TODO try results with separate edge type
    lft_edge = lft
    right_edge = right
    # num_labels = len(labels_dict)
    edge_types_num = 3
    # max_len, _ = x_target.shape
    # edges = numpy.zeros((max_len, max_len, edge_types_num))

    # edges[np.diag_indices_from(edges[:, :, slf]), slf] = 1
    # for i in range(edges.shape[0]):
    #     edges[i,i,slf] = 1

    edge_times = numpy.zeros((max_len, max_len, edge_types_num)) + float("inf")
    for i in range(edge_times.shape[0]):
        edge_times[i, i, slf] = i

    # labels = numpy.zeros((max_len, max_len, num_labels))
    label_times = numpy.zeros((max_len, max_len, num_labels)) + float("inf")
    token_num = 0
    idxs_stack = []
    tokens_stack = []
    label = False
    root = False
    edge_end = "@@|"
    for token_id, token in enumerate(x_target):
        # last = tokens_stack

        if token.endswith(edge_end):
            assert not label  # TODO gracefully convert, for inference time (just remove the assertion)
            label = True
            if token == ConllSent.REDUCE_L + edge_end:
                assert len(
                    idxs_stack) > 1, "tried to create a left edge with 1 word or less in the buffer" + str(idxs_stack)
                heads_ids, head_tokens = _last_word(idxs_stack, tokens_stack)
                idxs_stack, tokens_stack = idxs_stack[:-len(heads_ids)], tokens_stack[:-len(heads_ids)]
                dependent_ids, dependent_tokens = _last_word(idxs_stack, tokens_stack)
                idxs_stack, tokens_stack = idxs_stack[:-len(dependent_ids)], tokens_stack[:-len(dependent_ids)]
                idxs_stack += heads_ids
                tokens_stack += head_tokens
            elif token == ConllSent.REDUCE_R + edge_end:
                root = "root" in x_target[token_id + 1]
                dependent_ids, dependent_tokens = _last_word(idxs_stack, tokens_stack)
                idxs_stack, tokens_stack = idxs_stack[:-len(dependent_ids)], tokens_stack[:-len(dependent_ids)]
                heads_ids, head_tokens = _last_word(idxs_stack, tokens_stack, graceful=root)
            else:
                raise ValueError("Unexpected reduce action token" + token)

            # add edge
            for head in heads_ids:
                for dependent in dependent_ids:
                    edge_times[head, dependent, lft] = token_id + 1
                    edge_times[dependent, head, right] = token_id + 1

            # add a label to and from edge tokens
            for head in heads_ids:
                edge_times[token_id, head, lft_edge] = token_id + 1
            for dependent in heads_ids:
                edge_times[token_id, dependent, right_edge] = token_id + 1

        elif token.startswith("|@@"):
            assert label  # TODO gracefully convert, for inference time (make sure it works without the assertion)
            assert "root" not in token or root  # TODO gracefully convert, for inference time (make sure it works without the assertion)
            if label:
                label = False
                for head in heads_ids:
                    for dependent in dependent_ids:
                        label_times[head, dependent, labels_dict[token]] = token_id + 1

                # add a label to and from edges
                for head in heads_ids:
                    edge_times[token_id, head, lft_edge] = token_id + 1
                for dependent in heads_ids:
                    edge_times[token_id, dependent, right_edge] = token_id + 1

        else:
            assert not label  # TODO gracefully convert, for inference time (just remove the assertion)
            idxs_stack.append(token_id)
            tokens_stack.append(token)
            token_num += 1  # number of tokens that are not transitions
    return np.array(edge_times), np.array(label_times)