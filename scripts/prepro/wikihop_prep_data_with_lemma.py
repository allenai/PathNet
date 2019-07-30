import os
import json
from typing import List, Dict, Tuple
import re
from numpy import array
import time
from tqdm import tqdm
import argparse
from nltk import PorterStemmer

stemmer = PorterStemmer()


def load_dict(fname):
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data


def load_lines(fname):
    data = []
    with open(fname, 'r') as fp:
        for line in fp:
            data.append(json.loads(line))
    return data


def get_locs_given_objs(doc: str, word: str, objs: List):
    doctoks = doc.split()
    ch_locs = [ob.span()[0] for ob in objs]
    found_words = [ob.group(0) for ob in objs]
    assert len(found_words) == len(ch_locs)
    for fwidx, fw in enumerate(found_words):
        start_offset = len(fw) - len(fw.lstrip())
        ch_locs[fwidx] += start_offset
        found_words[fwidx] = fw.strip()
    widxs = get_widxs_from_chidxs(ch_locs, create_offsets(doctoks))
    locs = [(widx, widx + len(found_words[i].split()) - 1) for i, widx in enumerate(widxs)]
    return locs


def get_widxs_from_chidxs(chidxs: List[int],
                          offsets: List[List[int]]) -> List[int]:
    """
    Find word indices given character indices
    :param chidxs:
    :param offsets:
    :return:
    """
    last_ch_idx = offsets[0][0]
    assert max(chidxs) < offsets[-1][1] - last_ch_idx
    widxs = []
    for chidx in chidxs:
        for oi in range(len(offsets)):
            if chidx in range(offsets[oi][0] - last_ch_idx, offsets[oi][1] - last_ch_idx):
                widxs.append(oi)
                break
            elif chidx in range(offsets[oi][1] - last_ch_idx,
                                offsets[min(oi + 1, len(offsets))][0] - last_ch_idx):
                widxs.append(oi)
                break
    assert len(chidxs) == len(widxs)
    return widxs


def create_offsets(doctoks: List[str]) -> List[List[int]]:
    """
    create offsets for a document tokens
    :param doctoks:
    :return:
    """
    offsets = []
    char_count = 0
    for tok in doctoks:
        offsets.append([char_count, char_count + len(tok)])
        char_count = char_count + len(tok) + 1
    return offsets


def find_backup_path(docsents, q, cand, k=40):
    """
    Find a dummy backup path is no path could be found for a candidate
    :param docsents:
    :param q:
    :param cand:
    :param k:
    :return:
    """
    path_for_cand_dict = {"he_docidx": None,
                          "he_locs": None,
                          "e1wh_loc": None,
                          "e1_docidx": None,
                          "e1_locs": None,
                          "cand_docidx": None,
                          "cand_locs": None,
                          "he_words": ["BACKUP"],
                          "e1wh": "BACKUP",
                          "e1": "BACKUP",
                          "cand_words": ["BACKUP"]
                          }

    he = ' '.join(q[1:]).lower()
    if len(he.split()) == 0:
        path_for_cand_dict['he_docidx'] = 0
        path_for_cand_dict['he_locs'] = [(-1, -1)]
    else:
        pat_he = re.compile('(^|\W)' + re.escape(he) + '\W')

        for docssidx, docss in enumerate(docsents):
            doc = ' '.join(' '.join(sum(docss, [])).split())
            doc = doc.lower()
            he_objs = []
            for x in pat_he.finditer(doc):
                he_objs.append(x)
            if len(he_objs) > 0:
                path_for_cand_dict['he_docidx'] = docssidx
                path_for_cand_dict['he_locs'] = get_locs_given_objs(doc, he, he_objs)[:k]
                break

    cand = cand.lower()
    pat_cand = re.compile('(^|\W)' + re.escape(cand) + '\W')
    for docssidx, docss in enumerate(docsents):
        doc = ' '.join(' '.join(sum(docss, [])).split())
        doc = doc.lower()
        ca_objs = []
        for x in pat_cand.finditer(doc):
            ca_objs.append(x)
        if len(ca_objs) > 0:
            path_for_cand_dict['cand_docidx'] = docssidx
            path_for_cand_dict['cand_locs'] = get_locs_given_objs(doc, cand, ca_objs)[:k]
            break
    if path_for_cand_dict['he_docidx'] is None or path_for_cand_dict['he_locs'] is None:
        path_for_cand_dict['he_docidx'] = 0
        path_for_cand_dict['he_locs'] = [(-1, -1)]
    if path_for_cand_dict['cand_docidx'] is None or path_for_cand_dict['cand_locs'] is None:
        path_for_cand_dict['cand_docidx'] = 0
        path_for_cand_dict['cand_locs'] = [(0, 0)]

    return path_for_cand_dict


def get_min_abs_diff(list1: List[int], list2: List[int]) -> float:
    return float(min([min(abs(array(list1) - bi)) for bi in list2]))


def get_start_locs(list1: List[Tuple[int, int]]) -> List[int]:
    return [l[0] for l in list1]


def filter_paths(paths_for_cand: List[Dict], k=30) -> List[Dict]:
    """
    Scoring function for paths
    Keep only top-k paths

    path_for_cand_dict = {"he_docidx": he_docidx, #int
                          "he_locs": he_locs,  # List[int]
                          "e1wh_loc": e1_with_head_loc, # List[int]
                          "e1_docidx": e1_docidx, # int
                          "e1_locs": e1_locs, # List[int]
                          "cand_docidx": cand_docidx,  # int
                          "cand_locs": cand_locs  # int
                          }

    :param paths_for_cand:
    :param k:
    :return:
    """
    if len(paths_for_cand) < k:
        return paths_for_cand

    big_score = 1e7
    scores = []
    for path in paths_for_cand:
        he_docidx = path['he_docidx']
        he_locs = path['he_locs']
        e1wh_locs = path['e1wh_loc']
        # e1_docidx = path['e1_docidx']
        e1_locs = path['e1_locs']
        cand_docidx = path['cand_docidx']
        cand_locs = path['cand_locs']
        he_locs = get_start_locs(he_locs)
        e1wh_locs = get_start_locs(e1wh_locs)
        e1_locs = get_start_locs(e1_locs)
        cand_locs = get_start_locs(cand_locs)
        if he_docidx == cand_docidx:
            if get_min_abs_diff(he_locs, cand_locs) == 0:
                scores.append(big_score)
            else:
                scores.append(-1.0)
        else:
            he_e1wh_mindiff = get_min_abs_diff(he_locs, e1wh_locs)
            e1_cand_mindiff = get_min_abs_diff(cand_locs, e1_locs)
            if he_e1wh_mindiff == 0 or e1_cand_mindiff == 0:
                scores.append(big_score)
            else:
                scores.append(he_e1wh_mindiff + e1_cand_mindiff)
    assert len(scores) == len(paths_for_cand)

    temp = sorted(zip(paths_for_cand, scores),
                  key=lambda x: x[1],
                  reverse=False)  # ascending order
    sorted_paths, _ = map(list, zip(*temp))
    sorted_paths = sorted_paths[:k]

    return sorted_paths


def get_doc_len(docsents, docidx):
    doc = docsents[docidx]
    doc = ' '.join(sum(doc, [])).split()
    return len(doc)


def adjust_word_idxs(toks, widxs):
    split_toks = ' '.join(toks).split()
    if len(toks) == len(split_toks):
        return widxs
    else:
        mod_widxs = [widx - (len(toks[:widx]) - len(' '.join(toks[:widx]).split()))
                     for widx in widxs]
        return mod_widxs


def process_path_for_cand(path, docsents):
    """

    :param path:
    :param docsents:
    :return:
    """
    he_docidx = path['head_ent_docidx']
    he_doc_len = get_doc_len(docsents, he_docidx)
    he_ent_loc_dict = path['head_ent']
    he_locs = []
    he_words = list(he_ent_loc_dict.keys())
    for he_w in list(he_ent_loc_dict.keys()):
        start_loc_list = he_ent_loc_dict[he_w]
        start_loc_list = adjust_word_idxs(sum(docsents[he_docidx], []),
                                          start_loc_list)
        for s in start_loc_list:
            assert s < he_doc_len
        end_loc_list = [max(s, s + len(he_w.split()) - 1)
                        for s in start_loc_list]  # end is inclusive

        for e in end_loc_list:
            assert e < he_doc_len
        combined_locs = [(s, e) for s, e in zip(start_loc_list, end_loc_list)]
        he_locs += combined_locs

    e1 = path['e1']
    if e1 is None:
        e1_with_head_loc = [(-1, -1)]
        e1_docidx = None
        e1_locs = [(-1, -1)]
        cand_docidx = he_docidx
    else:
        e1wh_locs = [path['e1_with_head_widx']]
        e1wh_locs = adjust_word_idxs(sum(docsents[he_docidx], []), e1wh_locs)
        e1_with_head_loc = [(e1wh_locs[0],
                             e1wh_locs[0] + len(
                                 path['e1_with_head_ent'].split()) - 1)]  # inclusive end
        for s, e in e1_with_head_loc:
            assert s < he_doc_len
            assert e < he_doc_len
        e1_docidx = path['e1_docidx']
        e1_doc_len = get_doc_len(docsents, e1_docidx)
        e1_start_locs = path['e1_locs']
        e1_start_locs = adjust_word_idxs(sum(docsents[e1_docidx], []), e1_start_locs)
        e1_locs = [(e1s, e1s + len(e1.split()) - 1) for e1s in e1_start_locs]
        for s, e in e1_locs:
            assert s < e1_doc_len
            assert e < e1_doc_len
        cand_docidx = e1_docidx

    cand_loc_dict = path['cand_locs']
    cand_words = list(cand_loc_dict.keys())
    cand_doc_len = get_doc_len(docsents, cand_docidx)
    cand_locs = []
    for ca_w in list(cand_loc_dict.keys()):
        ca_w_start_loc_list = cand_loc_dict[ca_w]
        ca_w_start_loc_list = adjust_word_idxs(sum(docsents[cand_docidx], []),
                                               ca_w_start_loc_list)
        ca_w_end_loc_list = [s + len(ca_w.split()) - 1
                             for s in ca_w_start_loc_list]  # end is inclusive
        combined_locs_ca_w = [(s, e) for s, e in zip(ca_w_start_loc_list, ca_w_end_loc_list)]
        for s, e in combined_locs_ca_w:
            assert s < cand_doc_len
            assert e < cand_doc_len
        cand_locs += combined_locs_ca_w

    path_for_cand_dict = {"he_docidx": he_docidx,
                          "he_locs": he_locs,
                          "e1wh_loc": e1_with_head_loc,
                          "e1_docidx": e1_docidx,
                          "e1_locs": e1_locs,
                          "cand_docidx": cand_docidx,
                          "cand_locs": cand_locs,
                          "he_words": he_words,
                          "e1wh": path['e1_with_head_ent'],
                          "e1": e1,
                          "cand_words": cand_words
                          }
    return path_for_cand_dict


def process_allpaths_for_cand(cand, path_data_for_cand,
                              qtoks, docsents, max_num_paths):
    """
    process all paths for a particular candidate
    :param cand:
    :param path_data_for_cand:
    :param qtoks:
    :param docsents:
    :param max_num_paths:
    :return:
    """
    if len(qtoks) == 1 or len(path_data_for_cand) == 0:
        paths_for_cand = [find_backup_path(docsents, qtoks, cand)]
        return paths_for_cand

    paths_for_cand = []
    for pathforcand in path_data_for_cand:
        path_for_cand_dict_ = process_path_for_cand(pathforcand, docsents)
        paths_for_cand.append(path_for_cand_dict_)

    if len(paths_for_cand) == 0:
        paths_for_cand = [find_backup_path(docsents, qtoks, cand)]
    if len(paths_for_cand) > max_num_paths:
        paths_for_cand = filter_paths(paths_for_cand, k=max_num_paths)
    return paths_for_cand


def lemmatize_docsents(doc_sents: List[List[List[str]]]):
    doc_sents_lemma = []
    for idx, docl in enumerate(doc_sents):
        doc_sent = doc_sents[idx]
        docsent_lemma = []
        for senttoks in doc_sent:
            # startidx = len(sum(doc_sent[:sidx], []))
            docsent_lemma.append([stemmer.stem(tok) for tok in senttoks])
        doc_sents_lemma.append(docsent_lemma)
    return doc_sents_lemma


def process_paths(d: Dict, pathdata: Dict,
                  max_num_paths: int = 30,
                  lemmatize: bool = False) -> Dict:
    """
    Process paths
    :param d: data dictionary from extarcted paths
    :param pathdata:
    :param max_num_paths:
    :param lemmatize:
    :return:
    """
    # ans = ' '.join(d['answer'])
    docsents = d['docsents']  # List[List[List[List[str]]]
    qid = d['id']
    question = d['question']
    qpos = d['qpos']
    qner = d['qner']
    docpostags = d['docpostags']
    docners = d['docners']

    mod_docsents = []
    mod_docpostags = []
    mod_docners = []
    for docidx, doc in enumerate(docsents):
        sents = []
        sentpos = []
        sentner = []
        docpos = docpostags[docidx]
        docner = docners[docidx]
        for sentidx, sent in enumerate(doc):
            sents.append(' '.join(sent).split())
            stidx = len(sum(doc[:sentidx], []))
            endidx = stidx + len(' '.join(sent).split())
            sentpos.append(docpos[stidx:endidx])
            sentner.append(docner[stidx:endidx])
        mod_docsents.append(sents)
        mod_docpostags.append(sentpos)
        mod_docners.append(sentner)

    path_dict = {'id': qid, 'question': d['question'],
                 'qpos': qpos, 'qner': qner,
                 "docsents": mod_docsents,
                 "docpostags": mod_docpostags,
                 "docners": mod_docners,
                 'answer': d['answer'],
                 'candidates': d['candidates']}

    if lemmatize:
        docsents = lemmatize_docsents(docsents)
        question = [stemmer.stem(qw) for qw in question]

    # print(qid)
    assert len(list(pathdata['pathlist'].keys())) == len(d['candidates'])
    path_for_all_cands = []
    for cand in pathdata['pathlist'].keys():
        path_data_for_cand_ = pathdata['pathlist'][cand]
        paths_for_cand_ = process_allpaths_for_cand(cand, path_data_for_cand_,
                                                    question, docsents,
                                                    max_num_paths=max_num_paths)
        path_for_all_cands.append(paths_for_cand_)

    path_dict['paths'] = path_for_all_cands

    return path_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str,
                        help='Path to preprocessed data dir')
    parser.add_argument('pathdir', type=str,
                        help='Directory to read the paths')
    parser.add_argument('dumpdir', type=str,
                        help='Directory to dump the paths')
    parser.add_argument('--mode', type=str, default='dev',
                        help='Train/Dev')
    parser.add_argument('--maxnumpaths', type=int, default=30,
                        help='How many maximum paths to consider')
    parser.add_argument('--lemmatize', type=bool, default=False,
                        help='whether to lemmatize for path extraction')
    args = parser.parse_args()
    t0 = time.time()
    dumpdir = args.dumpdir
    if not os.path.isdir(args.dumpdir):
        os.makedirs(args.dumpdir)
    mode = args.mode
    max_num_paths = args.maxnumpaths
    lemmatize = args.lemmatize
    if mode == 'dev':
        dev_data = load_lines(args.datadir + '/dev-processed-spacy.txt')
        paths = load_lines(args.pathdir + '/dev-processed-spacy.txt.paths')
        print("All data loaded")

    with open(args.dumpdir + '/' + mode + '-path-lines.txt', 'w') as fp:
        if mode == "train":
            num_splits = 22  # check and modify how many splits were there for the train data paths
        else:
            num_splits = 1
        for sp in range(num_splits):
            if mode == 'train':
                dev_data = load_lines(args.datadir + '/train-split/split_' + str(sp) + '.json')
                paths = load_lines(args.pathdir + '/train-split/split_' + str(sp) + '.json.paths')
                print("Data loaded for split %d " % sp)
                assert len(dev_data) == len(paths)
            for dataidx, data in tqdm(enumerate(dev_data)):
                pathdata_ = paths[dataidx]
                path_dict_ = process_paths(data, pathdata_,
                                           max_num_paths=max_num_paths,
                                           lemmatize=lemmatize)
                fp.write(json.dumps(path_dict_) + '\n')

    print("Done!")
    print('Total time: %.4f (s)' % (time.time() - t0))


