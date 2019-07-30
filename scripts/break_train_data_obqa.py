import json
import sys


def load_examples(fpath):
    data = []
    with open(fpath, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))
    return data


def split_examples(data, dump_dir, bucket_size=500):
    num_split = int(len(data) / bucket_size)
    if num_split * bucket_size == len(data):
        splits = num_split
    else:
        splits = num_split + 1

    for i in range(splits):
        split_data = data[i * bucket_size: min((i + 1) * bucket_size, len(data))]
        with open(dump_dir + '/split_' + str(i) + '.json', 'w') as fp:
            for d in split_data:
                fp.write(json.dumps(d) + '\n')


data = load_examples(sys.argv[1])
print("data loaded")
split_examples(data, sys.argv[2])
print("done!")
