import argparse


def remove_edges(text):
    try:
        text = text.split()
    except AttributeError:
        pass
    res = []
    EDGE_END = "@@|"
    LABEL_START = "|@@"
    for word in text:
        if not (word.startswith(LABEL_START) or word.endswith(EDGE_END)):
            if (word.startswith("L") or word.startswith(
                    "R")) and EDGE_END in word:  # detokenization clasps ("." and other punctuation to the previous word)
                word = word[word.index(EDGE_END) + len(EDGE_END):]
            res.append(word)
    return " ".join(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='filename to process')
    parser.add_argument('--out', '-o', default=None, help='outfile')
    args = parser.parse_args()

    with open(args.filename) as fl:
        if args.out is None:
            for line in fl:
                print(remove_edges(line))
        else:
            with open(args.out, "w") as outfl:
                for line in fl:
                    outfl.write(remove_edges(line) + "\n")
