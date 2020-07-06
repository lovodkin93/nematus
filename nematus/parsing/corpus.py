"""
Convert text and conllu to trans(1) file.'
 conllu read based on https://github.com/iclementine/stackLSTM-parser/
"""

import conllu
import mmap
import re
import os
import pickle
# import torchtext


class ConllToken(object):
    """
    representation of a conll token
    """

    def __init__(self, id, form, lemma, upos, xpos, feats=None, head=None, deprel=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    @classmethod
    def from_string(cls, line):
        columns = [col if col !=
                   '_' else None for col in line.strip().split('\t')]
        columns[0] = int(columns[0])
        columns[6] = int(columns[6])
        id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = columns
        return cls(id, form, lemma, upos, xpos, feats, head, deprel, deps, misc)

    @classmethod
    def from_ordereddict(cls, token_dict):
        '''
        note: the representation for feats, misc, deps would be ordereddict instead of str,
        but it is okay, since we do not use them often
        '''
        return cls(*list(token_dict.values()))


class ConllSent(object):
    """
    an alternative format for representing conll-u sentences
    where each type of annotations is separately stored in its own list
    instead of a list of ConllTokens.
    """
    # SHIFT = 0
    # REDUCE_L = 1
    # REDUCE_R = 2

    SHIFT = "SHFT"
    REDUCE_L = "L"
    REDUCE_R = "R"

    def __init__(self, form, upos, xpos, head, deprel):
        self.form = list(form)
        self.upos = list(upos)
        self.xpos = list(xpos)
        self.head = list(head)
        self.deprel = list(deprel)
        self.transitions = None

    @classmethod
    def from_conllu(cls, sent):
        '''
        used to convert a list(OrderedDict) from conllu.parser() to this form
        '''
        sent_values = [x.values() for x in sent]
        fields = list(zip(*sent_values))
        if len(fields) < 8:
            print("Not enough fields:", fields, sent)
            for i in range(8 - len(fields)):
                fields.append([""] * len(fields[0]))
        return cls(fields[1], fields[3], fields[4], fields[6], fields[7])

    def arc_std(self):
        cursor = 0
        length = len(self.form)
        stack = []
        transitions = []

        def is_dependent():
            stack_top = stack[-1][0]
            for h in self.head[cursor:]:
                if h == stack_top:
                    return True
            return False

        while not (len(stack) == 1 and cursor == length):
            if len(stack) < 2:
                stack.append(
                    (cursor + 1, self.head[cursor], self.deprel[cursor]))
                transitions.append((self.SHIFT, self.form[cursor]))
                cursor += 1
            elif stack[-2][1] == stack[-1][0]:
                tok = stack.pop(-2)
                transitions.append((self.REDUCE_L, tok[2]))
            elif stack[-1][1] == stack[-2][0] and not is_dependent():
                tok = stack.pop()
                transitions.append((self.REDUCE_R, tok[2]))
            elif cursor < length:
                stack.append(
                    (cursor + 1, self.head[cursor], self.deprel[cursor]))
                transitions.append((self.SHIFT, self.form[cursor]))
                cursor += 1
            else:
                raise Exception("Not a valid projective tree.")
        tok = stack.pop()
        transitions.append((self.REDUCE_R, tok[2]))
        return transitions


class FastConlluReader(object):
    """
    Fast conll reader using mmap
    iter it and it yeilds list(ConllToken)
    """

    def __init__(self, fname, encoding='utf-8'):
        self.fname = fname
        self.f = open(self.fname, 'rb')
        self.encoding = encoding

    def __iter__(self):
        m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        sentence = list()
        line = m.readline()
        while line:
            pline = line.decode(self.encoding).strip()
            columns = pline.split('\t')
            line = m.readline()
            if pline == '':
                if len(sentence) > 0:
                    yield sentence
                sentence = list()
            elif pline[0] == '#' or '-' in columns[0] or '.' in columns[0]:
                continue
            else:
                sentence.append(ConllToken.from_string(pline))
        if len(sentence) > 0:
            yield sentence
        m.close()
        self.f.close()


class AltConlluReader(object):
    """
    Conllu reader which does not use mmap
    iter over it and it yields str, which represents a block of conllu sentence
    """

    def __init__(self, fname, encoding='utf-8'):
        self.fname = fname
        self.f = open(self.fname, 'rb')
        self.encoding = encoding

    def __iter__(self):
        m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        block = ''

        line = m.readline()
        while line:
            cur_line = line.decode(self.encoding)
            line = m.readline()
            if cur_line.strip() == '' or "# newdoc" in cur_line or "# newpar" in cur_line:
                if len(block):
                    sent = conllu.parse(block)[0]
                    sent = ConllSent.from_conllu(sent)
                    yield sent
                    block = ''
            else:
                block += cur_line
        if len(block):
            sent = conllu.parse(block)[0]
            sent = ConllSent.from_conllu(sent)
            yield sent
        m.close()
        self.f.close()

    def save_transitions(self, out_path=None):
        if out_path is None:
            out_path = os.path.splitext(self.fname)[0] + '_trans.pickle'

        proj_sents = []
        n_non_proj = 0
        with open(out_path + ".ids", 'w') as f:
            for i, sent in enumerate(self):
                try:
                    sent.transitions = sent.arc_std()
                    proj_sents.append(sent.transitions)
                    f.write(str(i) + "\n")
                except BaseException as e:
                    n_non_proj += 1
                    print("skipping non projective sentence number ", i)

        print("Skipping {} non-projective sentences".format(n_non_proj))
        with open(out_path, 'wb') as f:
            pickle.dump(proj_sents, f)

    def save(self, out_path=None):
        if out_path is None:
            out_path = os.path.splitext(self.fname)[0] + '.pickle'

        proj_sents = []
        n_non_proj = 0
        for sent in self:
            try:
                sent.transitions = sent.arc_std()
                proj_sents.append(sent)
            except:
                n_non_proj += 1

        print("Skipping {} non-projective sentences".format(n_non_proj))
        f = open(out_path, 'wb')
        pickle.dump(proj_sents, f)
        f.close()


# This corpus reader can be used when reading large text file into a memory can solve IO bottleneck of training.
# Use it exactly as the regular CorpusReader from the rnnlm.py
class FastCorpusReader(object):

    def __init__(self, fname):
        self.fname = fname
        self.f = open(fname, 'rb')

    def __iter__(self):
        # This usage of mmap is for a Linux\OS-X
        # For Windows replace prot=mmap.PROT_READ with access=mmap.ACCESS_READ
        m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        data = m.readline()
        while data:
            line = data
            data = m.readline()
            line = line.lower()
            line = line.strip().split()
            yield line


class CorpusReader(object):

    def __init__(self, fname, ids=None):
        """
        :param fname: file to read
        :param ids: line numbers to read, either a file separated by spaced or lines or a container (set, list etc.)
                    no input or false input (None, "", []) would result in reading all the file.
        """
        self.fname = fname
        if os.path.isfile(ids):
            with open(ids) as fl:
                self.ids = set((int(id) for id in fl.read().split()))
        else:
            self.ids = ids
        self.skipped = 0

    def __iter__(self):
        with open(self.fname) as fl:
            for i, line in enumerate(fl):
                if (not self.ids) or i in self.ids:
                    line = line.strip().split()
                    yield line
                else:
                    self.skipped += 1
        print("skipped", self.skipped, "sentences")


def convert_conllu_transitions(path, reader=FastConlluReader):
    reader = reader(path)
    for line in reader:
        print(line)
        print(line.arc_std())


def combine_bpe_transitions(bpe_path, trans_path, ids_path=None, split_actions=True):
    letters_pat = re.compile(r"[\W_]+")
    print(f"reading {trans_path}")
    try:
        with open(trans_path, "rb") as fl:
            transitions = pickle.load(fl)
    except pickle.UnpicklingError:
        transitions = CorpusReader(trans_path)
    if ids_path is None:
        ids_path = trans_path + ".ids"
    bpes = CorpusReader(bpe_path, ids_path)
    for line_num, (bpe_line, trans_line) in enumerate(zip(bpes, transitions)):
        line = []
        bpe_line = bpe_line
        bpe_iter = iter(bpe_line)
        for trans in trans_line:
            if trans[0] == ConllSent.SHIFT:
                word_bpes = []
                while True:
                    bpe = next(bpe_iter, None)
                    if bpe is None:
                        print("BPE and transitions mismatch length")
                    word_bpes.append(bpe)
                    if "@@" not in bpe:
                        break
                assert not set(letters_pat.sub('', "".join(word_bpes).lower().strip())) - set(trans[1].lower().strip()),\
                    "characters in bpe " + \
                    " ".join(word_bpes) + \
                    " and trans " + trans[1] + \
                    " don't match. Line " +str(line_num) +\
                    "Make sure tokenization " \
                    "is correct:\n" + " ".join(bpe_line) + \
                    "\n" + \
                    str(trans_line)
                line += word_bpes
            elif trans[0] == ConllSent.REDUCE_L or ConllSent.REDUCE_R:
                if split_actions:
                    line.append(trans[0] + "@@|")
                    line.append("|@@" + trans[1])
                else:
                    line.append(trans[0] + trans[1] + "@@|")
            else:
                raise NotImplementedError("Unknown transition", trans[0])

        yield line


def extract_text_from_combined_path(combined_path):
    combined_lines = CorpusReader(combined_path)
    for line in combined_lines:
        tokens = line.split()
        line = extract_text_from_combined_tokens(tokens)
        yield " ".join(line)


def extract_text_from_combined_tokens(tokens):
    text_tokens = []
    label = False
    for token in tokens:
        if token in [ConllSent.REDUCE_L + "@@|",  ConllSent.REDUCE_R + "@@|"]:
            assert label == False
            label = True
        elif label:
            assert token.startswith("|@@")
            label = False
        else:
            text_tokens.append(token)
    return text_tokens


def combine_conllu_bpe(conllu_path, bpe_path, trans_out_path, combined_path, split_actions=True, force=False, force_combine=False):
    if (not force) and (not force_combine) and os.path.isfile(combined_path):
        print("Skipping", combined_path, "file already exists.")
        return
    if not os.path.isfile(trans_out_path) or force:
        AltConlluReader(conllu_path).save_transitions(trans_out_path)
    # with open(trans_out, "rb") as fl:
    #     trans = pickle.load(fl)
    a = combine_bpe_transitions(bpe_path, trans_out_path, split_actions=split_actions)
    print(f"Writing to ", combined_path)
    with open(combined_path, "w") as fl:
        for line in a:
            fl.write(" ".join(line) + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert text and conllu to trans(1) file.')
    parser.add_argument('--force', '-f', action='store_true',
                        help='If set, replaces output file if exists')
    parser.add_argument('--force_combine', action='store_true',
                        help='If set, replaces output file if exists but does not recompute transitions')
    parser.add_argument('--split', '-s', action='store_true',
                        help='If set, transitions are split to edges and labels separately')
    parser.add_argument('--conllu', '-c', help='path to conllu')
    parser.add_argument('--bpe', '-b', help='path to bpe')
    parser.add_argument('--out', '-o', default="/cs/snapless/oabend/borgr/TG/en-de/output/",
                        help='path to output dir')

    args = parser.parse_args()

    # # path = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.conllu.en"
    # # path = "/cs/snapless/oabend/borgr/SSMT/data/UD/en_pud-ud-test_tmp.conllu"
    # path = "/cs/snapless/oabend/borgr/SSMT/data/UD/tmp.train.clean.unesc.tok.tc.conllu.en"
    # # bpe_path = "/cs/snapless/oabend/borgr/SSMT/data/UD/en_pud-ud-test_tmp.conllu.txt"
    # bpe_path = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en"
    # bpe_path = "/cs/snapless/oabend/borgr/SSMT/data/UD/tmp.train.clean.unesc.tok.tc.bpe.en"
    outdir_path = "/cs/snapless/oabend/borgr/TG/en-de/output/"

    force = args.force
    outdir_path = args.out
    path = args.conllu
    bpe_path = args.bpe
    split = args.split
    force_combine = args.force_combine
    if split:
        split_symb = ""
    else:
        split_symb = "1"
    trans_out = outdir_path + os.path.basename(path) + ".trns" + split_symb
    combined_path = outdir_path + os.path.splitext(os.path.basename(
        bpe_path))[0] + ".trns" + split_symb + os.path.splitext(os.path.basename(bpe_path))[1]
    # convert_conllu_transitions(path)
    print("Combining", path, "with bpe", bpe_path)
    combine_conllu_bpe(path, bpe_path, trans_out, combined_path, split_actions=split, force=force, force_combine=force_combine)
    print("main done")
