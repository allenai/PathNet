import re
from typing import List, Dict, Any, Tuple
from copy import deepcopy
import string

PUNKT_SET = set(string.punctuation)

VALIDNE_TAGS = ['PRODUCT', 'NORP', 'WORK_OF_ART',
                'LANGUAGE', 'LOC', 'GPE', 'PERSON',
                'FAC', 'ORG', 'EVENT']

VALIDPOS_TAGS = ['NN', 'NNP', 'NNPS', 'NNS']

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


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


def cluster_nertags(toks: List[str],
                    nertags: List[str]
                    ) -> (List[int], List[List[str]], List[List[str]]):
    """
    cluster based on ner tags
    """
    newtags = []
    newtoks = []
    startidxs = []

    tidx = 0
    while tidx < len(nertags):
        curtags = []
        curtoks = []
        curtag = nertags[tidx]
        curtok = toks[tidx]

        startidxs.append(tidx)
        curtags.append(curtag)
        curtoks.append(curtok)
        tidx += 1
        prevtag = curtag

        while True:
            if tidx < len(nertags) and prevtag in VALIDNE_TAGS:
                curtag = nertags[tidx]
                curtok = toks[tidx]
                if curtag == prevtag:
                    curtags.append(curtag)
                    curtoks.append(curtok)
                    tidx += 1
                else:
                    break
            else:
                break

        newtags.append(curtags)
        newtoks.append(curtoks)

    return startidxs, newtoks, newtags


def cluster_postags(toks: List[str],
                    postags: List[str]
                    ) -> (List[int], List[List[str]], List[List[str]]):
    """
    cluster based on postags
    """
    newtags = []
    newtoks = []
    startidxs = []

    tidx = 0
    while tidx < len(postags):
        curtags = []
        curtoks = []
        curtag = postags[tidx]
        curtok = toks[tidx]

        startidxs.append(tidx)
        curtags.append(curtag)
        curtoks.append(curtok)
        tidx += 1
        prevtag = curtag

        while True:
            if tidx < len(postags) and prevtag in VALIDPOS_TAGS:
                curtag = postags[tidx]
                curtok = toks[tidx]
                if (prevtag == 'NNP' and curtag == 'NNP') or \
                        (curtag == 'NN' or curtag == 'NNS' and prevtag == 'NN') or \
                        (prevtag == 'NNP' and curtag == 'NNS') or \
                        (prevtag == 'NNP' and curtag == 'NNS'):
                    curtags.append(curtag)
                    curtoks.append(curtok)
                    tidx += 1
                else:
                    break
            else:
                break

        newtags.append(curtags)
        newtoks.append(curtoks)

    return startidxs, newtoks, newtags


def get_all_ents(cwn: List[List[str]], cne: List[List[str]],
                 cwp: List[List[str]], cpo: List[List[str]],
                 neidxs: List[int], poidxs: List[int]) -> List[Tuple[int, str]]:
    """
    pick entities based on NER and POS tags
    :param cwn: clustered words based on NER
    :param cne: clustered NER
    :param cwp: clustered words based on POS
    :param cpo: clustered POS
    :param neidxs: start indices for NER clusters
    :param poidxs: start indices for POS clusters
    :return: set of other entities
    """
    entset = []
    for ne in VALIDNE_TAGS:
        for idx in range(len(cne)):
            if ne in cne[idx]:
                entset.append((neidxs[idx], cwn[idx]))

    for pos in VALIDPOS_TAGS:
        for idx in range(len(cpo)):
            if pos in cpo[idx]:
                entset.append((poidxs[idx], cwp[idx]))
    entset = set([(e[0], ' '.join(e[1])) for e in entset])
    uniq_entset = []
    for e in entset:
        if e not in uniq_entset:
            uniq_entset.append(e)
    uniq_entset = filter_trailing_puncts(uniq_entset)
    return uniq_entset


def filter_trailing_puncts(entity_list: List[Tuple[int, str]]):
    """
    filtering based on trailing punctuations
    :param entity_list: [(0, word1), (10, word2), ...]
    :return:
    """
    new_ent_list = []
    for eidx, e in enumerate(entity_list):
        e_toks = e[1].split(' ')
        if len(e_toks) == 1 and e_toks[0] in PUNKT_SET:
            continue
        else:
            if e_toks[0] in PUNKT_SET and e_toks[-1] in PUNKT_SET:
                new_ent_list.append((e[0] + 1, ' '.join(e_toks[1:-1])))
            elif e_toks[0] in PUNKT_SET:
                new_ent_list.append((e[0] + 1, ' '.join(e_toks[1:])))
            else:
                new_ent_list.append(e)
    return new_ent_list


def find_word_re(toks: List[str],
                 w: str) -> (bool, Any):
    """
    find word in a doc using re
    :param toks:
    :param w:
    :return:
    """
    try:
        pat = re.compile(w.lower() + '\W')
    except:
        return False, None

    objs = []
    doc_str = ' '.join(toks).lower()
    for x in pat.finditer(doc_str):
        objs.append(x)
    if len(objs) > 0:
        # found_words = [ob.group(0).strip() for ob in objs]
        start_ch_idxs = [ob.span()[0] for ob in objs]
        start_widxs = [len(doc_str[:chidx].split()) for chidx in start_ch_idxs]
        for i in range(len(start_widxs)):
            if start_widxs[i] > len(toks) - 1:
                start_widxs[i] = len(toks) - 1
        return True, start_widxs
    else:
        return False, None


def get_locations(doctoks: List[str], word: str,
                  docoffsets: List[List[int]],
                  allow_partial: bool = False) -> Dict:
    """
    check the locations. if word = x; find X
    if word = XY; find X*Y
    if word = XYZ; find x*Z
    :param doctoks:
    :param word:
    :param docoffsets:
    :param allow_partial:
    :return:
    """
    assert len(doctoks) == len(docoffsets)
    wordspanss = {}
    word = word.lower()
    try:
        re.compile(word)
    except:
        return wordspanss

    w_toks = word.split()
    if len(w_toks) < 1:
        return wordspanss

    if len(w_toks) == 1:
        pat = re.compile('(^|\W)' + re.escape(word) + '\W')
    else:
        if allow_partial:
            pat = re.compile('((^|\W)' + re.escape(word) + '\W|(^|\W)' + re.escape(w_toks[0]) +
                             '\W(.{,40}?\W)??' + re.escape(w_toks[-1]) + '\W)')
        else:
            pat = re.compile('(^|\W)' + re.escape(word) + '\W')

    objs = []
    doc_str = ' '.join(doctoks).lower()
    for x in pat.finditer(doc_str):
        objs.append(x)
    if len(objs) > 0:
        found_words = [ob.group(0) for ob in objs]
        start_ch_idxs = [ob.span()[0] for ob in objs]
        assert len(found_words) == len(start_ch_idxs)
        for fwidx, fw in enumerate(found_words):
            start_offset = len(fw) - len(fw.lstrip())
            start_ch_idxs[fwidx] += start_offset
            found_words[fwidx] = fw.strip()
        start_widxs = get_widxs_from_chidxs(start_ch_idxs, deepcopy(docoffsets))
        for swidx in range(len(start_widxs)):
            if start_widxs[swidx] > len(doctoks) - 1:
                start_widxs[swidx] = len(doctoks) - 1
        for i, w in enumerate(found_words):
            if w not in list(wordspanss.keys()):
                wordspanss[w] = [start_widxs[i]]
            else:
                wordspanss[w].append(start_widxs[i])
    return wordspanss


def find_sentidx(doctoks: List[List[str]],
                 word_idx: int) -> int:
    """
    find the sentidx given word idx
    :param doctoks:
    :param word_idx:
    :return:
    """
    count = 0
    for idx, doc in enumerate(doctoks):
        count += len(doc)
        if word_idx < count:
            return idx
    return len(doctoks) - 1


def remove_overlapping_ents(he_loc_dict: Dict,
                            ent_list: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    remove the overlapping entities (potential duplicates)
    :param he_loc_dict:
    :param ent_list:
    :return:
    """

    def is_present(idx: int, spans: List[Tuple[int, int]]):
        for sp in spans:
            if idx in range(sp[0], sp[1]):
                return True
        return False

    he_span_dict = {}
    for key in list(he_loc_dict.keys()):
        # exclusive end
        he_span_dict[key] = [(he_loc_dict[key][i],
                              he_loc_dict[key][i] + len(key.split(' ')))
                             for i in range(len(he_loc_dict[key]))]
    he_all_spans = sum(he_span_dict.values(), [])
    ent_list_spans = [(e[0], e[0] + len(e[1].split(' ')), e[1]) for e in ent_list]
    new_ent_list = []
    for eidx, e in enumerate(ent_list_spans):
        if is_present(e[0], he_all_spans) or is_present(e[1], he_all_spans):
            continue
        else:
            new_ent_list.append((e[0], e[2]))
    return new_ent_list


def unroll(counts: List[int], l: List[Any]) -> List[List[Any]]:
    counts = [0] + counts
    unrolled_list = []
    for idx in range(len(counts) - 1):
        curr_idx = sum(counts[:idx + 1])
        next_idx = curr_idx + counts[idx + 1]
        unrolled_list.append(l[curr_idx:next_idx])
    return unrolled_list


def get_non_stop_words(toks: List[str]):
    """
    retrieve non-stopwords
    :param toks:
    :return:
    """
    lw = []
    for tok in toks:
        tok = tok.lower()
        if tok not in STOPWORDS and tok not in PUNKT_SET:
            lw.append(tok)
    if len(lw) == 0:
        lw = [toks[0]]
    return lw


def get_lookup_words(toks: List[str],
                     pos: List[str], ner: List[str],
                     type='nonstopwords') -> List[str]:
    """
    get the potential (head) entities when (head) entity
    is not given specifically. This is necessary for
    OBQA like settings
    :param toks: question/candidate
    :param pos: postags
    :param ner: ners
    :param type: nonstopwords/noun phrases,ners(nps)
    :return:
    """
    if type == 'nonstopwords':
        lw = get_non_stop_words(toks)
    elif type == 'nps':
        neidxs, cwn, cne = cluster_nertags(toks, ner)
        poidxs, cwp, cpo = cluster_postags(toks, pos)
        lw = get_all_ents(cwn, cne, cwp, cpo, neidxs, poidxs)
        lw = [tup[1] for tup in lw]
        if len(lw) == 0:
            # fall back to non stopwords
            lw = get_non_stop_words(toks)
    else:
        raise NotImplementedError
    return lw


def get_locations_words(doctoks: List[str], words: List[str],
                        docoffsets: List[List[int]],
                        allow_partial: bool = False) -> Dict:
    """
    get locations for the words
    :param doctoks:
    :param words: list of words
    :param docoffsets:
    :param allow_partial:
    :return:
    """
    wordspans = {}
    for word in words:
        ws = get_locations(doctoks, word, docoffsets,
                           allow_partial=allow_partial)
        if len(ws) > 0:
            for key, values in ws.items():
                if key in wordspans:
                    wordspans[key] += values
                else:
                    wordspans[key] = values
    return wordspans


def lemmatize_docsents(doc_sents, stem):
    """
    lemmatize the document sentences
    :param doc_sents:
    :param stem:
    :return:
    """
    doc_sents_lemma = []
    for idx, docl in enumerate(doc_sents):
        doc_sent = doc_sents[idx]
        docsent_lemma = []
        for senttoks in doc_sent:
            docsent_lemma.append([stem(tok) for tok in senttoks])
        doc_sents_lemma.append(docsent_lemma)
    return doc_sents_lemma
