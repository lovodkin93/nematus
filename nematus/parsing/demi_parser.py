import random
import os


def get_dependencies(nums):
    return _get_dependencies(list(range(1, nums + 1)), 0)


def _get_dependencies(sublist, top_head):
    length = len(sublist)
    if length == 1:
        return [top_head]
    elif length == 0:
        return []
    else:
        right_length = length // 2
        left_length= (length - 1) // 2
        assert  left_length + right_length + 1 == length
        return _get_dependencies(sublist[:left_length], sublist[left_length]) + [top_head] + _get_dependencies(sublist[-right_length:], sublist[left_length])


def parse_lines(line):
    words = line.split()
    pointers = get_dependencies(len(words))
    labels = ["det"]
    labels = random.choices(labels, k=len(pointers))
    lines = [(i+1, word, word, label, label, "_", pointer, label, label, "_") for i, (word, pointer, label) in enumerate(zip(words, pointers, labels))]
    return lines

def demi_parse_lines(line):
    lines = parse_lines(line)
    lines = ["\t".join((str(characteristic) for characteristic in characteristics)) for characteristics in lines]
    lines = "\n".join(lines) + "\n"
    return lines

if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Convert text and conllu to trans(1) file.')
    # parser.add_argument('--force', '-f', action='store_true',
    #                     help='If set, replaces output file if exists')
    # parser.add_argument('--in', '-i', help='path to sentences file, one sentence per line')
    # parser.add_argument('--out', '-o', default="/cs/snapless/oabend/borgr/TG/en-de/output/",
    #                     help='path to output dir')
    #
    # args = parser.parse_args()
    #
    # # # path = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.conllu.en"
    # # # path = "/cs/snapless/oabend/borgr/SSMT/data/UD/en_pud-ud-test_tmp.conllu"
    # # path = "/cs/snapless/oabend/borgr/SSMT/data/UD/tmp.train.clean.unesc.tok.tc.conllu.en"
    # # # bpe_path = "/cs/snapless/oabend/borgr/SSMT/data/UD/en_pud-ud-test_tmp.conllu.txt"
    # # bpe_path = "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en"
    # # bpe_path = "/cs/snapless/oabend/borgr/SSMT/data/UD/tmp.train.clean.unesc.tok.tc.bpe.en"
    # outdir_path = "/cs/snapless/oabend/borgr/TG/en-de/output/"

    # force = args.force
    # outdir_path = args.out
    # in_path = args.in
    # out_path = args.out
    # split = args.split
    in_path = "/cs/snapless/oabend/borgr/SSMT/data/UD/tmp.txt"
    out_path = "/cs/snapless/oabend/borgr/SSMT/data/UD/"
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        out_path = os.path.join(out_path, os.path.splitext(os.path.basename(in_path))[0]) + ".dconllu"
    with open(out_path, "w") as out_fl:
        out_fl.write("# newdoc id = demi01001" + "\n")
        with open(in_path) as in_fl:
            for i, line in enumerate(in_fl):
                out_fl.write("# sent_id = " + str(i) + "\n")
                out_fl.write("# text = " + line.strip() + "\n")
                out_fl.writelines(demi_parse_lines(line))
                out_fl.write("\n")
    print("main done")
