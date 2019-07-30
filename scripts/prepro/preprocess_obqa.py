"""Preprocess the OBQA data."""

import sys
import argparse
import os
from typing import List, Dict, Any

import json
import time
import random

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
sys.path.append("./")
from pathnet.tokenizers.spacy_tokenizer import SpacyTokenizer

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None
ANNTOTORS = {'lemma', 'pos', 'ner'}


def init():
    global TOK
    TOK = SpacyTokenizer(annotators=ANNTOTORS)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text: str) -> Dict:
    """Call the global process tokenizer
    on the input text.
    """
    global TOK
    tokens = TOK.tokenize(text)
    words = tokens.words()
    output = {
        'words': words,
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
        'sentences': [words],
    }
    return output


def splittext(text: str) -> Dict:
    output = {
        'words': text.strip().split()
    }
    return output


# ------------------------------------------------------------------------------
# Process data examples
# ------------------------------------------------------------------------------


def load_dataset(path: str,
                 shuffle_docs: bool = False) -> Dict:
    """Load json file and store
    fields separately.
    """
    with open(path) as f:
        data = json.load(f)
    output = {'qids': [], 'questions': [], 'answers': [],
              'contextlists': [], 'candidatelists': []}
    for ex in data:
        output['qids'].append(ex['id'])
        output['questions'].append(' '.join(ex['query'].split()))
        if "answer" in ex:
            output['answers'].append(' '.join(ex['answer'].split()))
        else:
            output['answers'].append("DUMMYANSWER")
        if shuffle_docs:
            random.shuffle(ex['supports'])
        supports = [' '.join(s.split()) for s in ex['supports']]
        output['contextlists'].append(supports)
        candidates = [' '.join(c.split()) for c in ex['candidates']]
        output['candidatelists'].append(candidates)
    return output


def unroll(counts: List[int], l: List[Any]) -> List[List[Any]]:
    counts = [0] + counts
    unrolled_list = []
    for idx in range(len(counts) - 1):
        curr_idx = sum(counts[:idx + 1])
        next_idx = curr_idx + counts[idx + 1]
        unrolled_list.append(l[curr_idx:next_idx])
    return unrolled_list


def process_dataset(data: Dict, num_workers: int = None):
    """Iterate processing (tokenize, parse, etc)
    data multi-threaded.
    """
    if num_workers > 1:
        make_pool = partial(Pool, num_workers, initializer=init)
        workers = make_pool(initargs=())

        q_tokens = workers.map(tokenize, data['questions'])
        workers.close()
        workers.join()
    else:
        init()
        q_tokens = [tokenize(q) for q in data['questions']]

    # documents are in list format
    dcounts = [len(c) for c in data['contextlists']]
    if num_workers > 1:
        workers = make_pool(initargs=())
        c_tokens = workers.map(tokenize, sum(data['contextlists'], []))
        workers.close()
        workers.join()
    else:
        print("Tokenizing docs without multiprocessing..")
        c_tokens = [tokenize(c) for c in sum(data['contextlists'], [])]
    context_tokens = unroll(dcounts, c_tokens)

    if "answers" in data:
        if num_workers > 1:
            workers = make_pool(initargs=())
            ans_tokens = workers.map(tokenize, data['answers'])
            workers.close()
            workers.join()
        else:
            ans_tokens = [tokenize(a) for a in data['answers']]
    else:
        ans_tokens = None

    candcounts = [len(c) for c in data['candidatelists']]
    if num_workers > 1:
        workers = make_pool(initargs=())
        cnd_tokens = workers.map(tokenize, sum(data['candidatelists'], []))
        workers.close()
        workers.join()
    else:
        print("Tokenizing candidates without multiprocessing..")
        cnd_tokens = [tokenize(ca) for ca in data['candidatelists']]
    cand_tokens = unroll(candcounts, cnd_tokens)

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        qpos = q_tokens[idx]['pos']
        qner = q_tokens[idx]['ner']

        # supporting documents are in list format
        documents = [c['words'] for c in context_tokens[idx]]
        offsets = [c['offsets'] for c in context_tokens[idx]]
        cpostags = [c['pos'] for c in context_tokens[idx]]
        cners = [c['ner'] for c in context_tokens[idx]]
        clemmas = [c['lemma'] for c in context_tokens[idx]]
        doc_sents = [c['sentences'] for c in context_tokens[idx]]

        if ans_tokens is not None:
            answer = ans_tokens[idx]['words']
        else:
            answer = ["DUMMYANSWER"]

        candidates = [ca['words'] for ca in cand_tokens[idx]]
        candidatelemmas = [ca['lemma'] for ca in cand_tokens[idx]]
        candidatepos = [ca['pos'] for ca in cand_tokens[idx]]
        candidatener = [ca['ner'] for ca in cand_tokens[idx]]

        yield {
            'id': data['qids'][idx],
            'question': question,
            'qlemma': qlemma,
            'qpos': qpos,
            'qner': qner,
            'documents': documents,
            'offsets': offsets,
            'docsents': doc_sents,
            'docpostags': cpostags,
            'docners': cners,
            'doclemmas': clemmas,
            'answer': answer,
            'candidates': candidates,
            'candidatelemmas': candidatelemmas,
            'candidatepos': candidatepos,
            'candidatener': candidatener
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='Path to data file')
    parser.add_argument('out_dir', type=str, help='Path to output file dir')
    parser.add_argument('--split', type=str, help='Filename for train/dev split')
    parser.add_argument('--num-workers', type=int, default=6)
    args = parser.parse_args()

    t0 = time.time()

    in_file = args.data_file  # os.path.join(args.data_dir, args.split + '.json')
    print('Loading data %s' % in_file, file=sys.stderr)
    dataset = load_dataset(in_file)

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    out_file = os.path.join(
        args.out_dir, '%s-processed-%s.txt' % (args.split, 'spacy')
    )
    print('Will write to file %s' % out_file, file=sys.stderr)
    with open(out_file, 'w') as f:
        for ex in process_dataset(dataset,  # args.tokenizer,
                                  args.num_workers):
            f.write(json.dumps(ex) + '\n')
    print('Total time: %.4f (s)' % (time.time() - t0))
