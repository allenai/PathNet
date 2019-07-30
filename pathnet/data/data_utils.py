import numpy
import re
import json
from copy import deepcopy
from numpy import array
from typing import List, Tuple, Dict, Any
from nltk import PorterStemmer

stemmer = PorterStemmer()

POSTAGS = ['UNKPOSTAG', '', 'VBP', '.', 'MD', 'SYM', 'VBD', 'POS', 'NNS', 'VBZ',
           'PRP$', 'IN', '``', 'NN', 'WP', 'VBG', "''", 'TO', 'PRP',
           '-RRB-', 'LS', 'JJR', 'ADD', '$', 'UH', 'JJS', 'WP$', 'AFX',
           'NNPS', 'VB', 'CD', 'DT', ':', ',', 'VBN', '_SP', '-LRB-', 'EX',
           'RBS', 'WDT', 'FW', 'HYPH', 'PDT', 'RB', 'RP', 'CC', 'WRB', 'JJ',
           'NFP', 'RBR', 'XX', 'NNP']
NERTAGS = ['UNKNERTAG', '', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE',
           'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT',
           'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']


def extract_ent_reps(docreps: List[array],
                     docidx: int, locs: List[Tuple[int, int]],
                     aggregate: str = 'max') -> array:
    """
    :param docreps: N x T x H
    :param docidx:
    :param locs:
    :param aggregate:
    :return:
    """
    hdim = docreps[0].shape[-1]  # H
    combined_locs = []
    for loc in locs:
        if loc[0] is None or loc[0] == -1:
            # x = numpy.zeros(hdim)
            # y = numpy.zeros(hdim)
            combined = numpy.zeros(hdim)
        else:
            # x = docreps[docidx][loc[0]]
            # y = docreps[docidx][loc[1]]
            combined = numpy.mean(docreps[docidx][loc[0]:loc[1] + 1, :], 0)
            if numpy.isnan(combined).any():
                combined = numpy.zeros(hdim)
        # x, y format (TODO for other func types)
        # combined = numpy.concatenate((x, y))
        combined_locs.append(combined)
    combined_locs = array(combined_locs)
    if aggregate == 'max':
        return numpy.max(combined_locs, 0)
    else:
        raise NotImplementedError
        # return None


def pack_span_idxs(span_locs: List[List[List[Tuple[int, int]]]]):
    """
    packing of span indices
    :param span_locs: C * P * L * 2
    :return:
    """
    all_span_locs = sum(sum(span_locs, []), [])  # CPL * 2
    loc_tracks = []
    for cidx, c in enumerate(span_locs):
        for pidx, p in enumerate(c):
            for lidx, l in enumerate(p):
                loc_tracks.append([cidx, pidx, lidx])
    return all_span_locs, loc_tracks


def pack_doc_idxs(p_list: List[List[int]],
                  span_locs: List[List[List[Tuple[int, int]]]]) -> List[int]:
    """
    packing of document indices
    :param p_list:
    :param span_locs:
    :return:
    """
    p_list_loc_rptd = []
    for cidx, c in enumerate(span_locs):
        for pidx, p in enumerate(c):
            p_list_loc_rptd += [p_list[cidx][pidx] for _ in range(len(p))]
    return p_list_loc_rptd


def find_cand_locs_fromdoclist(docs: List[str],
                               choice_text_list: List[str],
                               lowercase: bool = True):
    """
    find candidate locations from a given document list
    :param docs:
    :param choice_text_list:
    :param lowercase:
    :return:
    """
    offsets = [create_offsets(doc.split()) for doc in docs]
    if lowercase:
        docs = [' '.join(doc.split()).lower() for doc in docs]
    else:
        docs = [' '.join(doc.split()) for doc in docs]
    all_docidxs = []
    all_spans = []

    for choice_id, choice_text in enumerate(choice_text_list):
        # find the choice locations in doc
        if lowercase:
            choice_text = choice_text.lower()
        objs = []
        docidxs = []
        try:
            pat = re.compile('(^|\W)' + choice_text + '\W')
            for didx, doc in enumerate(docs):
                doc_objs = []
                for x in pat.finditer(doc):
                    doc_objs.append(x)
                objs += doc_objs
                docidxs += [didx] * len(doc_objs)
        except:
            print(f"Could not compile candidate {choice_text}")

        # might want to initialize with [(-1, -1)] as there will be path scores to avoid nan
        choice_spans = [(-1, -1)]  # [(-1, -1)]
        found_words = [ob.group(0) for ob in objs]
        if len(objs) > 0:
            choice_ch_spans = [ob.span()[0] for ob in objs]
            assert len(found_words) == len(choice_ch_spans)
            widxs = []
            for fwidx, fw in enumerate(found_words):
                start_offset = len(fw) - len(fw.lstrip())
                choice_ch_spans[fwidx] += start_offset
                found_words[fwidx] = fw.strip()
                widx = get_widxs_from_chidxs([choice_ch_spans[fwidx]],
                                             deepcopy(offsets[docidxs[fwidx]]))[0]
                widxs.append(widx)

            # widxs = get_widxs_from_chidxs(choice_ch_spans, deepcopy(offsets))
            choice_spans = [(widxs[i], widxs[i] + len(found_words[i].split()) - 1)
                            for i in range(len(widxs))]
        if len(docidxs) == 0:
            docidxs = [0]
        assert len(choice_spans) == len(docidxs)
        all_spans.append(choice_spans)
        all_docidxs.append(docidxs)
    return all_spans, all_docidxs


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


def get_locs_forall(paths, doc_sent_boundary_dict,
                    num_word_shift_dict, mod_docsents):
    """
    get locations for corresponding to all the paths
    :param paths:
    :param doc_sent_boundary_dict:
    :param num_word_shift_dict:
    :param mod_docsents:
    :return:
    """
    num_docs = len(mod_docsents)
    p1_list: List[List[int]] = []  # cands * num_paths
    p2_list: List[List[int]] = []

    he_locs_list, e1wh_locs_list = [], []  # C * P * L * 2
    e1_locs_list, ca_locs_list = [], []

    for cidx in range(len(paths)):
        p1s = [paths[cidx][i]['he_docidx'] if paths[cidx][i]['he_docidx'] is not None else 0
               for i in range(len(paths[cidx]))]
        p2s = [paths[cidx][i]['cand_docidx'] if paths[cidx][i]['cand_docidx'] is not None
               else num_docs - 1
               for i in range(len(paths[cidx]))]
        p1_list.append(p1s)
        p2_list.append(p2s)

        he_locs = [paths[cidx][i]['he_locs'] if paths[cidx][i]['he_locs'] is not None
                   else [(-1, -1)]
                   for i in range(len(paths[cidx]))]
        e1wh_locs = [paths[cidx][i]['e1wh_loc'] if paths[cidx][i]['e1wh_loc'] is not None
                     else [(-1, -1)]
                     for i in range(len(paths[cidx]))]
        e1_locs = [paths[cidx][i]['e1_locs'] if paths[cidx][i]['e1_locs'] is not None
                   else [(-1, -1)]
                   for i in range(len(paths[cidx]))]
        ca_locs = [paths[cidx][i]['cand_locs'] if paths[cidx][i]['cand_locs'] is not None
                   else [(-1, -1)]
                   for i in range(len(paths[cidx]))]

        if doc_sent_boundary_dict is not None and num_word_shift_dict is not None:
            for pidx in range(len(paths[cidx])):
                for locidx in range(len(he_locs[pidx])):
                    if he_locs[pidx][locidx][0] != -1:
                        he_locs[pidx][locidx] = (he_locs[pidx][locidx][0] -
                                                 num_word_shift_dict[p1s[pidx]],
                                                 he_locs[pidx][locidx][1] -
                                                 num_word_shift_dict[p1s[pidx]])
                        if he_locs[pidx][locidx][0] > len(sum(mod_docsents[p1s[pidx]], [])) - 1:
                            he_locs[pidx][locidx] = (len(sum(mod_docsents[p1s[pidx]], [])) - 1,
                                                     len(sum(mod_docsents[p1s[pidx]], [])) - 1)

                for locidx in range(len(e1wh_locs[pidx])):
                    if e1wh_locs[pidx][locidx][0] != -1:
                        e1wh_locs[pidx][locidx] = (
                            e1wh_locs[pidx][locidx][0] - num_word_shift_dict[p1s[pidx]],
                            e1wh_locs[pidx][locidx][1] - num_word_shift_dict[p1s[pidx]])
                        if e1wh_locs[pidx][locidx][0] > len(sum(mod_docsents[p1s[pidx]], [])) - 1:
                            e1wh_locs[pidx][locidx] = (len(sum(mod_docsents[p1s[pidx]], [])) - 1,
                                                       len(sum(mod_docsents[p1s[pidx]], [])) - 1)

                for locidx in range(len(e1_locs[pidx])):
                    if e1_locs[pidx][locidx][0] != -1:
                        e1_locs[pidx][locidx] = (e1_locs[pidx][locidx][0] -
                                                 num_word_shift_dict[p2s[pidx]],
                                                 e1_locs[pidx][locidx][1] -
                                                 num_word_shift_dict[p2s[pidx]])
                        if e1_locs[pidx][locidx][0] > len(sum(mod_docsents[p2s[pidx]], [])) - 1:
                            e1_locs[pidx][locidx] = (len(sum(mod_docsents[p2s[pidx]], [])) - 1,
                                                     len(sum(mod_docsents[p2s[pidx]], [])) - 1)

                for locidx in range(len(ca_locs[pidx])):
                    if ca_locs[pidx][locidx][0] != -1:
                        ca_locs[pidx][locidx] = (ca_locs[pidx][locidx][0] -
                                                 num_word_shift_dict[p2s[pidx]],
                                                 ca_locs[pidx][locidx][1] -
                                                 num_word_shift_dict[p2s[pidx]])
                        if ca_locs[pidx][locidx][0] > len(sum(mod_docsents[p2s[pidx]], [])) - 1:
                            ca_locs[pidx][locidx] = (len(sum(mod_docsents[p2s[pidx]], [])) - 1,
                                                     len(sum(mod_docsents[p2s[pidx]], [])) - 1)

        he_locs_list.append(he_locs)
        e1wh_locs_list.append(e1wh_locs)
        e1_locs_list.append(e1_locs)
        ca_locs_list.append(ca_locs)

    return p1_list, p2_list, he_locs_list, e1wh_locs_list, e1_locs_list, ca_locs_list


def get_max_locs(paths, he_locs_list, e1wh_locs_list,
                 e1_locs_list, ca_locs_list):
    """
    obtaining the maximum number of locations
    :param paths:
    :param he_locs_list:
    :param e1wh_locs_list:
    :param e1_locs_list:
    :param ca_locs_list:
    :return:
    """
    max_he_locs = 0
    max_e1wh_locs = 0
    max_e1_locs = 0
    max_ca_locs = 0
    for cidx in range(len(paths)):
        for pidx in range(len(paths[cidx])):
            num_he_locs = len(he_locs_list[cidx][pidx])
            max_he_locs = max(max_he_locs, num_he_locs)

            num_e1wh_locs = len(e1wh_locs_list[cidx][pidx])
            max_e1wh_locs = max(max_e1wh_locs, num_e1wh_locs)

            num_e1_locs = len(e1_locs_list[cidx][pidx])
            max_e1_locs = max(max_e1_locs, num_e1_locs)

            num_ca_locs = len(ca_locs_list[cidx][pidx])
            max_ca_locs = max(max_ca_locs, num_ca_locs)

    return max_he_locs, max_e1wh_locs, max_e1_locs, max_ca_locs
