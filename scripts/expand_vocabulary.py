"""
Mainly required for evaluating on blind test set
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Dict

import torch
from allennlp.models.archival import CONFIG_NAME
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.common import Tqdm
from allennlp.data import DatasetReader
from allennlp.models import load_archive, archive_model
sys.path.append("./")
import pathnet

logger = logging.getLogger('scripts.expand_vocabulary')
logger.setLevel(logging.INFO)


def main(file, embeddings, model, emb_wt_key, namespace, output_dir):
    archive = load_archive(model)
    config = archive.config
    os.makedirs(output_dir, exist_ok=True)
    config.to_file(os.path.join(output_dir, CONFIG_NAME))

    model = archive.model
    # first expand the vocabulary
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    instances = dataset_reader.read(file)
    vocab = model.vocab

    # get all the tokens in the new file
    namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for instance in Tqdm.tqdm(instances):
        instance.count_vocab_items(namespace_token_counts)
    old_token_size = vocab.get_vocab_size(namespace)
    print("Before expansion: Number of instances in {} namespace: {}".format(namespace,
                                                                             old_token_size))
    if namespace not in namespace_token_counts:
        logger.error("No tokens found for namespace: {} in the new input file".format(namespace))
    # identify the new tokens in the new instances
    token_to_add = set()
    token_hits = 0
    for token, count in namespace_token_counts[namespace].items():
        if token not in vocab._token_to_index[namespace]:
            # new token, must add
            token_to_add.add(token)
        else:
            token_hits += 1
    print("Found {} existing tokens and {} new tokens in {}".format(token_hits,
                                                                    len(token_to_add), file))

    # add the new tokens to the vocab
    for token in token_to_add:
        vocab.add_token_to_namespace(token=token, namespace=namespace)
    archived_parameters = dict(model.named_parameters())

    # second, expand the embedding matrix
    for name, weights in archived_parameters.items():
        # find the wt matrix for the embeddings
        if name == emb_wt_key:
            if weights.dim() != 2:
                logger.error("Expected an embedding matrix for the parameter: {} instead"
                             "found {} tensor".format(emb_wt_key, weights.shape))
            emb_dim = weights.shape[-1]
            print("Before expansion: Size of emb matrix: {}".format(weights.shape))
            # Loading embeddings for old and new tokens since that is cleaner than copying all
            # the embedding loading logic here
            all_embeddings = _read_pretrained_embeddings_file(embeddings, emb_dim,
                                                              vocab, namespace)
            # concatenate the new entries i.e last token_to_add embeddings to the original weights
            if len(token_to_add) > 0:
                weights.data = torch.cat([weights.data, all_embeddings[-len(token_to_add):, :]])
            print("After expansion: Size of emb matrix: {}".format(weights.shape))

    # save the files needed by the model archiver
    model_path = os.path.join(output_dir, "weight.th")
    model_state = model.state_dict()
    torch.save(model_state, model_path)
    vocab.save_to_files(os.path.join(output_dir, "vocabulary"))
    archive_model(output_dir, weights="weight.th")

    # more debug messages
    new_token_size = vocab.get_vocab_size(namespace)
    for name, weights in archived_parameters.items():
        if name == emb_wt_key:
            print("Size of emb matrix: {}".format(weights.shape))
    print("After expansion: Number of instances in {} namespace: {}".format(namespace,
                                                                            new_token_size))


if __name__ == "__main__":
    """
    Usage:
    python -u scripts/expand_vocabulary.py \
         --file <path to the test file in the same format as expected by the dataset reader> \
         --emb_wt_key _text_field_embedder.token_embedder_tokens.weight \
         --embeddings <path to Glove file> \
         --model <path to your trained model.tar.gz> \
         --output_dir <directory where the expanded model file will be saved>
    """
    parser = argparse.ArgumentParser(description='Expand vocabulary (and embeddings) of a model '
                                                 'based on a new file')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the new file (should be readable by the model\'s '
                             'dataset reader')
    parser.add_argument('--emb_wt_key', type=str, required=True,
                        help='Parameter name for the token embedding weight matrix')
    parser.add_argument('--namespace', type=str, default="tokens", help='Namespace to expand')
    parser.add_argument('--embeddings', type=str, required=True, help='Path to the embeddings file')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--output_dir', type=str, help='The output directory to store the '
                                                       'final model')

    args = parser.parse_args()
    main(args.file, args.embeddings, args.model, args.emb_wt_key, args.namespace, args.output_dir)