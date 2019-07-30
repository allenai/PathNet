import json
import argparse


def get_out_json(predf, outf):
    """
    :param predf:
    :param outf:
    :return:
    """
    preddict = {}
    with open(predf, 'r') as fp:
        for line in fp:
            data = json.loads(line)
            preddict[data["id"]] = data["answer"]

    with open(outf, 'w') as fp:
        json.dump(preddict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predfile', type=str,
                        help='Path to prdicted file')
    parser.add_argument('outfile', type=str,
                        help='Output file')
    args = parser.parse_args()
    get_out_json(args.predfile, args.outfile)
