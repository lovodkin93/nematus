import argparse
def replace_by_shifts(text):
    try:
        text = text.split()
    except AttributeError:
        pass
    res = ["SHIFT" if "@@" not in word else word for word in text]
    return " ".join(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='filename to process')
    parser.add_argument('--out', '-o', default=None, help='outfile')
    args = parser.parse_args()

    with open(args.filename) as fl:
        if args.out is None:
            for line in fl:
                print(replace_by_shifts(text))
        else:
            with open(args.out) as outfl:
                for line in fl:
                    outfl.write(replace_by_shifts(text) + "\n")