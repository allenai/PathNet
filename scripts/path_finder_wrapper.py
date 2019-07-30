from typing import List, Dict, Optional
from nltk import PorterStemmer

import sys
sys.path.append("./")
from pathnet.tokenizers.spacy_tokenizer import SpacyTokenizer
from pathnet.pathfinder.path_extractor import PathFinder
from pathnet.pathfinder.obqa_path_extractor import ObqaPathFinder
from pathnet.pathfinder.util import lemmatize_docsents

stemmer = PorterStemmer()
stem = stemmer.stem

ANNTOTORS = {'lemma', 'pos', 'ner'}
TOK = SpacyTokenizer(annotators=ANNTOTORS)


def tokenize(text: str) -> Dict:
    """Call the global process tokenizer
    on the input text.
    """
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
        'sentences': tokens.sentences(),
    }
    return output


def process_data(documents: List[str], question: str, candidate: str) -> Dict:
    """
    process the instance
    :param documents: list of documents
    :param question:
    :param candidate:
    :return:
    """
    data = dict()
    q_tokens = tokenize(question)
    data['question'] = q_tokens['words']
    data['qlemma'] = q_tokens['lemma']
    data['qpos'] = q_tokens['pos']
    data['qner'] = q_tokens['ner']
    cand_tokens = tokenize(candidate)
    data['candidate'] = cand_tokens['words']
    data['cpos'] = cand_tokens['lemma']
    data['cner'] = cand_tokens['ner']
    data['clemma'] = cand_tokens['lemma']
    doc_tokens = [tokenize(doc) for doc in documents]
    data['documents'] = [doc['words'] for doc in doc_tokens]
    data['docners'] = [doc['ner'] for doc in doc_tokens]
    data['docpostags'] = [doc['pos'] for doc in doc_tokens]
    data['docsents'] = [doc['sentences'] for doc in doc_tokens]
    return data


def find_paths(documents: List[str], question: str, candidate: str,
               style='wikihop') -> Optional[List]:
    """
    Get the list of paths for a given (documents, question, candidate)
    :param documents: list of documents
    :param question:
    :param candidate:
    :param style: "wikihop" or "OBQA" -- OBQA style is for plain text questions
    :return:
    """
    sentlimit = 1
    nearest_only = False
    d = process_data(documents, question, candidate)

    doc_ners = d['docners']
    doc_postags = d['docpostags']
    doc_sents = d['docsents']

    qpos = d["qpos"]
    qner = d["qner"]
    qlemma = d['qlemma']
    rel = qlemma[0]
    entity = ' '.join(qlemma[1:]).lower()
    candidates = []
    orig_candidates = [d['candidate']]
    for ctoks in orig_candidates:
        sctoks = [stemmer.stem(ca) for ca in ctoks]
        if sctoks in candidates:
            candidates.append(ctoks)
        else:
            candidates.append(sctoks)
    candidates = [' '.join(cand) for cand in candidates]
    candpos = [d['cpos']]
    candner = [d['cner']]

    doc_sents_lemma = lemmatize_docsents(doc_sents, stem)

    if style.strip().lower() == "wikihop":
        pf = PathFinder("qid", doc_sents_lemma,
                        entity, rel,
                        candidates,
                        answer=None,
                        sentlimit=sentlimit,
                        nearest_only=nearest_only)
    else:
        pf = ObqaPathFinder("qid", doc_sents_lemma,
                            qlemma, qpos, qner,
                            candidates, candpos, candner,
                            answer=None, sentlimit=sentlimit,
                            nearest_only=nearest_only)

    paths = pf.get_paths(doc_ners, doc_postags)
    if len(paths) == 0:
        print("No Paths Found !!")
        return None
    # pathdict = {"id": "qid", "pathlist": paths[list(paths.keys())[0]]}
    return paths[list(paths.keys())[0]]
