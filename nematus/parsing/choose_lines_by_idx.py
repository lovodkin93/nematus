
def main(sentences_path, output_path, idx_path):
    print("Taking subset of lines")
    with open(sentences_path, encoding="utf-8") as sentences_fl:
        with open(output_path, "w", encoding="utf-8") as out_fl:
            with open(idx_path) as idx_fl:
                sentence_idx = 0
                for idx in idx_fl:
                    idx = int(idx)
                    while sentence_idx <= idx:
                        sentence_idx += 1
                        sentence = sentences_fl.readline()
                    out_fl.write(sentence)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert text and conllu to trans(1) file.')
    parser.add_argument('--sentences', '-s', help='path to sentences')
    parser.add_argument('--idx', '-i', help='path to idx')
    parser.add_argument('--out', '-o', help='path to output file')

    args = parser.parse_args()
    main(args.sentences, args.out, args.idx)