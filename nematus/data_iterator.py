import sys
import numpy
import logging

import gzip

from django.contrib.sitemaps.views import x_robots_tag
from parsing.corpus import extract_text_from_combined_tokens
from parsing.corpus import ConllSent
from util import reset_dict_vals
import numpy as np
import subprocess
import ast

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .util import load_dict, parse_transitions
    from . import shuffle
except (ModuleNotFoundError, ImportError) as e:
    from util import load_dict, parse_transitions
    import shuffle


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding="UTF-8")
    return open(filename, mode, encoding="UTF-8")


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
                 preprocess_script=None,
                 target_graph=False,
                 target_labels_num=None,
                 splitted_action=False,
                 ignore_empty=False,
                 same_scene_masks=None):
        self.preprocess_script = preprocess_script
        self.source_orig = source
        self.target_orig = target
        self.ignore_empty = ignore_empty
        self.same_scene_masks_orig=same_scene_masks
        if self.preprocess_script:
            logging.info("Executing external preprocessing script...")
            proc = subprocess.Popen(self.preprocess_script)
            proc.wait()
            logging.info("done")
        if keep_data_in_memory:
            self.source, self.target = FileWrapper(source), FileWrapper(target)
            if same_scene_masks:
                self.same_scene_masks = FileWrapper(same_scene_masks)
            else:
                self.same_scene_masks = None
            if shuffle_each_epoch:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
                if same_scene_masks:
                    self.same_scene_masks.shuffle_lines(r)
        elif shuffle_each_epoch:
            #logging.info("AVIVSL11: source is {0} and target is {1} and masks is {2}".format(source,target, same_scene_masks))
            self.source_orig = source
            self.target_orig = target
            self.same_scene_masks_orig = same_scene_masks
            if same_scene_masks:
                self.source, self.target, self.same_scene_masks = shuffle.jointly_shuffle_files(
                    [self.source_orig, self.target_orig, self.same_scene_masks_orig], temporary=True)
            else:
                self.source, self.target = shuffle.jointly_shuffle_files(
                    [self.source_orig, self.target_orig], temporary=True)
                self.same_scene_masks = None
            #logging.info("AVIVSL12: source is {0} and target is {1} and masks is {2}".format(self.source, self.target, self.same_scene_masks))
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
            if same_scene_masks:
                self.same_scene_masks = fopen(same_scene_masks, 'r')
            else:
                self.same_scene_masks = None
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict, model_type))
        self.target_dict = load_dict(target_dict, model_type)
        self.target_graph = target_graph
        self.target_labels_num = target_labels_num
        if self.target_graph:
            self.splitted_action = splitted_action
            self.target_actions, self.target_labels = parse_transitions(self.target_dict, self.splitted_action)
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
        self.same_scene_masks_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def set_remove_parse(self, bool):
        self.remove_parse = bool

    def reset(self):
        if self.preprocess_script:
            logging.info("Executing external preprocessing script...")
            proc = subprocess.Popen(self.preprocess_script)
            proc.wait()
            logging.info("done")
            if self.keep_data_in_memory:
                self.source, self.target = FileWrapper(self.source_orig), FileWrapper(self.target_orig)
                self.same_scene_masks = FileWrapper(self.same_scene_masks_orig) if self.same_scene_masks_orig else None
            else:
                self.source = fopen(self.source_orig, 'r')
                self.target = fopen(self.target_orig, 'r')
                self.same_scene_masks = fopen(self.same_scene_masks_orig, 'r') if self.same_scene_masks_orig else None
        if self.shuffle:
            if self.keep_data_in_memory:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
                if self.same_scene_masks:
                    self.same_scene_masks.shuffle_lines(r)
            else:
                if self.same_scene_masks_orig:
                    self.source, self.target, self.same_scene_masks = shuffle.jointly_shuffle_files(
                        [self.source_orig, self.target_orig, self.same_scene_masks_orig], temporary=True)
                else:
                    self.source, self.target = shuffle.jointly_shuffle_files(
                        [self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)
            if self.same_scene_masks:
                self.same_scene_masks.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        # same_scene_mask = [] #TODO: AVIVSL - changed

        longest_source = 0
        longest_target = 0

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(
            self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()
                ssm = self.same_scene_masks.readline() if self.same_scene_masks_orig else None #same_scene_mask
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                if self.remove_parse:
                    ss = extract_text_from_combined_tokens(ss)
                    tt = extract_text_from_combined_tokens(tt)
                if (ss and tt) or not self.ignore_empty:
                    self.source_buffer.append(ss)
                    self.target_buffer.append(tt)
                    if self.same_scene_masks_orig:
                           self.same_scene_masks_buffer.append(ast.literal_eval(ssm)) # ast.literal_eval translates a string represting a list of lists into a list of lists
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
                _ssmbuf = [self.same_scene_masks_buffer[i] for i in tidx] if self.same_scene_masks_orig else None
                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
                self.same_scene_masks_buffer = _ssmbuf
            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
                if self.same_scene_masks_orig:
                    self.same_scene_masks_buffer.reverse()

        def lookup_token(t, d, unk_val):
            return d[t] if t in d else unk_val

        try:
            # actual work here
            while True:

                # read from source file and map to words index
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
                if self.same_scene_masks_orig:
                    ssm = self.same_scene_masks_buffer.pop()
                    source.append((ss_indices, ssm))
                else:
                    ssm = None
                    source.append(ss_indices)

                # read from source file and map to words index
                tt = self.target_buffer.pop()

                tt_indices = [lookup_token(w, self.target_dict,
                                           self.target_unk_val) for w in tt]
                if self.target_vocab_size != None:
                    tt_indices = [w if w < self.target_vocab_size
                                  else self.target_unk_val
                                  for w in tt_indices]

                if self.target_graph:
                    # logging.info("read" + str(tt))
                    tt_edge_time, tt_label_time, parents_time = convert_text_to_graph(
                        ["<GO>"] + tt, self.maxlen + 1, self.target_labels, self.target_labels_num,
                        split=self.splitted_action)
                    target.append((tt_indices, tt_edge_time, tt_label_time, parents_time))

                else:
                    target.append(tt_indices)

                # read from same_scene_mask file
                # if self.same_scene_masks_orig:
                #     ssm = self.same_scene_masks_buffer.pop()
                #     same_scene_mask.append(ssm)

                longest_source = max(longest_source, len(ss_indices))
                longest_target = max(longest_target, len(tt_indices))

                if self.token_batch_size:
                    if len(source) * longest_source > self.token_batch_size or \
                            len(target) * longest_target > self.token_batch_size:
                        # remove last sentence pair (that made batch over-long)
                        source.pop() # for same_scene_mask, source includes a tuple of (source sentence, same_scene mask)
                        target.pop()
                        self.source_buffer.append(ss) #but the ss is just the sentence, without the same_scene mask
                        self.target_buffer.append(tt)
                        if self.same_scene_masks_orig:
                            # same_scene_mask.pop()
                            self.same_scene_masks_buffer.append(ssm)
                        break

                else:
                    if len(source) >= self.batch_size or \
                            len(target) >= self.batch_size:
                        break
        except IOError:
            self.end_of_data = True
        # if not self.same_scene_masks_orig: # make sure same_scene_mask is of same length as source (so iterator will work) even when not used
        #     same_scene_mask = [None] * len(source)
        return source, target


def _last_word(idxs, tokens,
               graceful=False):
    if not idxs:
        if not graceful:
            raise IndexError("Asked for last words on an empty buffer")
        return [], []
    i = -1
    while -i < len(idxs):
        i -= 1
        if not tokens[i].endswith("@@"):
            i += 1
            break
    return idxs[i:], tokens[i:]


def convert_text_to_graph(x_target, max_len, labels_dict, num_labels, split=False, attend_max=False, graceful=False):
    # TODO document
    # TODO connect reduce and labels to the popped out
    # TODO convert to work with indexes
    slf = 0
    lft = 1
    right = 2
    # TODO try results with separate edge type
    lft_edge = lft
    right_edge = right
    edge_types_num = 3
    edge_times = numpy.zeros((max_len, max_len, edge_types_num)) + float("inf")
    for i in range(edge_times.shape[0]):
        edge_times[i, i, slf] = i
    if attend_max:
        sentence_len = max_len
    else:
        sentence_len = len(x_target)
    parents = numpy.zeros((sentence_len, sentence_len), dtype=np.float32) + float("inf")
    for i in range(parents.shape[0]):
        parents[i, i] = i
    if num_labels:
        label_times = numpy.zeros((max_len, max_len, num_labels)) + float("inf")
    token_num = 0
    idxs_stack = []
    tokens_stack = []
    label = False
    root = False
    edge_end = "@@|"
    label_start = "|@@"
    # if ("L@@|" not in x_target):
    #     raise
    for token_id, token in enumerate(x_target):
        # last = tokens_stack

        if token.endswith(edge_end):
            assert graceful or not label or not num_labels, token + "," + " ".join(x_target)
            label = True
            if split:
                raise ValueError("why")
            if token.startswith(ConllSent.REDUCE_L):
                min_len_cond = len(idxs_stack) > 1
                if graceful and not min_len_cond:
                    heads_ids = []
                    continue
                assert min_len_cond, "tried to create a left edge with 1 words or less in the buffer " + x_target
                heads_ids, head_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_stack, tokens_stack = idxs_stack[:-len(heads_ids)], tokens_stack[:-len(heads_ids)]
                dependent_ids, dependent_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_stack, tokens_stack = idxs_stack[:-len(dependent_ids)], tokens_stack[:-len(dependent_ids)]
                idxs_stack += heads_ids
                tokens_stack += head_tokens
            elif token.startswith(ConllSent.REDUCE_R):
                if split:
                    label_token = x_target[token_id + 1]
                else:
                    label_token = token
                    assert label_token == x_target[token_id], (label_token, token_id, x_target)
                root = graceful or "root" in label_token  # Root expects only one words to exist
                dependent_ids, dependent_tokens = _last_word(idxs_stack, tokens_stack, graceful=graceful)
                idxs_stack, tokens_stack = idxs_stack[:-len(dependent_ids)], tokens_stack[:-len(dependent_ids)]
                heads_ids, head_tokens = _last_word(idxs_stack, tokens_stack, graceful=root)
            else:
                raise ValueError("Unexpected reduce action token" + token)

            # add parents
            for head in heads_ids:
                parents[token_id, head] = token_id
                for dependent in dependent_ids:
                    parents[token_id, dependent] = token_id
                    parents[dependent, head] = token_id
                    parents[dependent, token_id] = token_id

            # add edge
            for head in heads_ids:
                for dependent in dependent_ids:
                    edge_times[head, dependent, lft] = token_id
                    edge_times[dependent, head, right] = token_id

            # add a label to and from edge tokens
            for head in heads_ids:
                edge_times[token_id, head, lft_edge] = token_id
            for dependent in dependent_ids:
                edge_times[token_id, dependent, right_edge] = token_id

            if split:
                assert token.endswith(edge_end)
            elif num_labels:
                if token not in labels_dict or labels_dict[token] >= label_times.shape[-1]:
                    logging.error(
                        f"Token error, token: {token} labels_dict:{labels_dict} label_times shape {label_times.shape}")
                    logging.error("token num:", labels_dict[token])
                for head in heads_ids:
                    for dependent in dependent_ids:
                        label_times[head, dependent, labels_dict[token]] = token_id
                label = False

        elif token.startswith(label_start):
            assert not split
            assert graceful or label or not num_labels
            assert graceful or "root" not in token or root
            if label and num_labels:
                label = False
                if token not in labels_dict or labels_dict[token] >= label_times.shape[-1]:
                    logging.error(
                        f"Token error, token: {token} labels_dict:{labels_dict} label_times shape {label_times.shape}")
                    logging.error("token num:", labels_dict[token])
                for head in heads_ids:
                    for dependent in dependent_ids:
                        label_times[head, dependent, labels_dict[token]] = token_id

                # add label relevance to parents
                # add parents
                for head in heads_ids:
                    parents[token_id, head] = token_id
                    for dependent in dependent_ids:
                        parents[token_id, dependent] = token_id
                        parents[dependent, token_id] = token_id

                # add a label to and from edges
                for head in heads_ids:
                    edge_times[token_id, head, lft_edge] = token_id
                for dependent in dependent_ids:
                    edge_times[token_id, dependent, right_edge] = token_id

        else:
            assert graceful or not label or not num_labels, token + "," + " ".join(x_target)
            label = False
            idxs_stack.append(token_id)
            tokens_stack.append(token)
            token_num += 1  # number of tokens that are not transitions (actual subwords)
    # print("edge array", np.array(edge_times, dtype=np.float32))
    # print("x_target for graph convert", x_target, np.array(x_target).shape)

    edge_times = np.array(edge_times, dtype=np.float32)
    if num_labels:
        label_times = np.array(label_times, dtype=np.float32)
    else:
        label_times = np.zeros(1, dtype=np.float32)
    # print("initial parents", parents, parents.shape)
    # print("initial parents non_inf", np.argwhere(parents < 1000), parents[parents < 1000])
    # print("initial shapes", edge_times.shape, label_times.shape, parents.shape)
    return edge_times, label_times, parents
