import json
import sys


def load_dict(fname):
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data


def dump_dict(fname, data):
    with open(fname, 'w') as fp:
        json.dump(data, fp)


def break_data(data, fname, bucket_size=5000):
    num_buckets = len(data) / bucket_size
    if num_buckets > int(num_buckets):
        num_buckets = int(num_buckets) + 1
    else:
        num_buckets = int(num_buckets)
    for i in range(num_buckets):
        print("Bucket: ", i)
        data_bucket = data[i * bucket_size:min((i + 1) * bucket_size, len(data))]
        dump_dict(fname[:-5] + str(i) + ".json", data_bucket)


if __name__ == "__main__":
    data = load_dict(sys.argv[1])
    break_data(data, sys.argv[1])
