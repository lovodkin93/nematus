
def main(sentences_path, output_path, idx_path):
    print("Taking subset of lines")
    sentences_fl = open(sentences_path, encoding="utf-8")
    sentence_idx = 0
    with open(output_path, "w", encoding="utf-8") as out_fl:
        with open(idx_path) as idx_fl:
            for idx in idx_fl:
                if idx:
                    idx = int(idx)
                    # read file from beginning if id is larger than previous
                    if sentence_idx > idx:
                        sentences_fl.close()
                        sentences_fl = open(sentences_path, encoding="utf-8")
                        sentence_idx = 0

                    while sentence_idx <= idx:
                        sentence_idx += 1
                        sentence = sentences_fl.readline()
                    out_fl.write(sentence)
    sentences_fl.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='subsample lines by index file (without reading all file to memory).')
    parser.add_argument('--sentences', '-s', help='path to sentences')
    parser.add_argument('--idx', '-i', help='path to idx')
    parser.add_argument('--out', '-o', help='path to output file')

    args = parser.parse_args()
    main(args.sentences, args.out, args.idx)