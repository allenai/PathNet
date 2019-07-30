"""
Script for the pasth finding step for WikiHop
"""

import os
import json
import argparse
import time
from typing import List, Dict, Any
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial
from nltk import PorterStemmer

import sys
sys.path.append("./")
from pathnet.pathfinder.path_extractor import PathFinder

stemmer = PorterStemmer()


def load_examples(fpath: str) -> List[Dict]:
    """Load the preprocessed examples
    """
    data = []
    with open(fpath, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))
    return data


def init():
    pass


def process_examples(d: Dict) -> Dict:
    lemma = True
    sentlimit = 1
    nearest_only = False
    qid = d['id']
    if "answer" in d:
        ans = ' '.join(d['answer']).lower()
    else:
        ans = 'DUMMYANSWER'
    doc_ners = d['docners']
    doc_postags = d['docpostags']
    doc_sents = d['docsents']

    if not lemma:
        rel = d['question'][0]
        entity = ' '.join(d['question'][1:]).lower()
        candidates = [' '.join(cand) for cand in d['candidates']]
        pf = PathFinder(qid, doc_sents,
                        entity, rel,
                        candidates,
                        answer=ans,
                        sentlimit=sentlimit,
                        nearest_only=nearest_only)
    else:
        qlemma = [stemmer.stem(qtok) for qtok in d['question']]
        rel = qlemma[0]
        entity = ' '.join(qlemma[1:]).lower()
        candidates = []
        orig_candidates = d['candidates']
        for ctoks in orig_candidates:
            sctoks = [stemmer.stem(ca) for ca in ctoks]
            if sctoks in candidates:
                candidates.append(ctoks)
            else:
                candidates.append(sctoks)
        candidates = [' '.join(cand) for cand in candidates]
        doc_sents_lemma = []
        for idx, docl in enumerate(doc_sents):
            doc_sent = doc_sents[idx]
            docsent_lemma = []
            for senttoks in doc_sent:
                docsent_lemma.append([stemmer.stem(tok) for tok in senttoks])
            doc_sents_lemma.append(docsent_lemma)

        pf = PathFinder(qid, doc_sents_lemma,
                        entity, rel,
                        candidates,
                        answer=ans,
                        sentlimit=sentlimit,
                        nearest_only=nearest_only)

    paths = pf.get_paths(doc_ners, doc_postags)
    pathdict = {"id": qid, "pathlist": paths}
    return pathdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str,
                        help='Path to preprocessed data file')
    parser.add_argument('dumpdir', type=str,
                        help='Directory to dump the paths')
    parser.add_argument('--sentlimit', type=int, default=1,
                        help='how many next sentences to look for ne/nouns')
    parser.add_argument('--take_nearest_only', type=bool, default=False,
                        help='whether to take nearest candidate only')
    parser.add_argument('--numworkers', type=int, default=6,
                        help='number of workers for multiprocessing')
    args = parser.parse_args()

    t0 = time.time()

    infile = args.datafile
    data = load_examples(infile)
    print("Data Loaded..")

    num_paths = 0
    num_cands = 0

    print("Computing paths..")
    workers = args.numworkers
    make_pool = partial(Pool, workers, initializer=init)

    workers = make_pool(initargs=())
    path_list = tqdm(workers.map(process_examples, data), total=len(data))
    workers.close()
    workers.join()

    print("Analysing stats..")
    max_paths_per_cand = 0
    max_paths_per_q = 0
    for ps in path_list:
        paths_per_q = 0
        for p in ps['pathlist'].values():
            num_paths += len(p)
            num_cands += 1
            paths_per_q += len(p)
            if len(p) > max_paths_per_cand:
                max_paths_per_cand = len(p)
        if paths_per_q > max_paths_per_q:
            max_paths_per_q = paths_per_q

    if not os.path.isdir(args.dumpdir):
        os.makedirs(args.dumpdir)
    with open(os.path.join(args.dumpdir, os.path.basename(infile) + '.paths'), 'w') as fp:
        for pp in path_list:
            fp.write(json.dumps(pp) + '\n')

    print("Avg #paths/question: %.4f" % (num_paths / len(data)))
    print("Avg #paths/candidate: %.4f" % (num_paths / num_cands))
    print("Max #paths/question: %d" % max_paths_per_q)
    print("Max #paths/candidate: %d" % max_paths_per_cand)
    print('Total time: %.4f (s)' % (time.time() - t0))
