import argparse
def remove_edges(text):
    try:
        text = text.split()
    except AttributeError:
        pass
    res = [word for word in text if not (word.startswith("|@@") or word.endswith("@@|"))]
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
            with open(args.out) as outfl:
                for line in fl:
                    outfl.write(remove_edges(line) + "\n")