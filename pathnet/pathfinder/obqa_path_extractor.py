from typing import List, Dict, Any, Tuple
from copy import deepcopy

from pathnet.pathfinder.util import create_offsets, get_widxs_from_chidxs, get_locations, \
    cluster_postags, cluster_nertags, get_all_ents, find_sentidx, remove_overlapping_ents, \
    get_lookup_words, get_locations_words


class ObqaPathFinder(object):
    """class for path finder for the OBQA dataset"""

    def __init__(self, qid, docsentlist: List[List[List[str]]],
                 # entity: str, relation: str,
                 question: List[str],
                 qpos: List[str],
                 qner: List[str],
                 candidates: List[str],
                 candpos: List[List[str]],
                 candner: List[List[str]],
                 answer: str = None,
                 sentlimit: int = 1, minentlen: int = 3,
                 nearest_only: bool = False,
                 qenttype="nps", candenttype="nps") -> None:
        self.docsentlist = docsentlist
        self.docoffsets = []
        for docsent in self.docsentlist:
            doctoks = sum(docsent, [])
            self.docoffsets.append(create_offsets(doctoks))

        self.qenttype = qenttype
        self.candenttype = candenttype
        self.question = question
        self.qpos = qpos
        self.qner = qner
        self.entity_set = get_lookup_words(self.question, self.qpos,
                                           self.qner, type=self.qenttype)

        self.candidates = candidates
        self.candpos = candpos
        self.candner = candner
        self.answer = answer
        self.sentlimit = sentlimit
        self.minentlen = minentlen
        self.nearest_only = nearest_only
        self.id = qid

    def find_entity_in_all_docs(self, entity_list: List[str]) -> (int, List[List[Any]]):
        num_locs = 0
        alldoc_he_locs = []
        for docidx, doc in enumerate(self.docsentlist):
            doc_he_locs = []
            for sentidx, sent in enumerate(doc):
                prev_num_words = len(sum(doc[:sentidx], []))
                offsets_for_sent = self.docoffsets[docidx][prev_num_words:
                                                           prev_num_words + len(sent)]
                he_locs = {}
                for entity in entity_list:
                    cur_entity_locs = get_locations(sent, entity,
                                                    offsets_for_sent,
                                                    allow_partial=True)  # {w1: [start_ind_1, ...], w2: [start_ind_1, ], ...}
                    if len(cur_entity_locs) > 0:
                        for key, values in cur_entity_locs.items():
                            if key not in he_locs.keys():
                                he_locs[key] = values
                            else:
                                he_locs[key] += values
                doc_he_locs.append(he_locs)
                num_locs += len(he_locs)
            alldoc_he_locs.append(doc_he_locs)
        return num_locs, alldoc_he_locs

    def get_all_he_locs(self) -> List[List[Any]]:
        """
        get all possible head entity locations
        :return:
        """
        num_locs, alldoc_he_locs = self.find_entity_in_all_docs(self.entity_set)
        if num_locs > 0:
            return alldoc_he_locs
        entity_list = []
        for entity in self.entity_set:
            if "-" in entity:
                entity_list.append(entity.replace('-', ' - '))
            if " of " in entity:
                entity_list.append(entity.split(" of ")[0].strip())
            if ":" in entity:
                entity_list.append(entity.split(":")[0].strip())
        num_locs, alldoc_he_locs = self.find_entity_in_all_docs(entity_list)
        if num_locs > 0:
            return alldoc_he_locs
        entity_list = []
        for entity in self.entity_set:
            if entity[-1:] == 's':
                entity_list.append(entity[:-1])
        num_locs, alldoc_he_locs = self.find_entity_in_all_docs(entity_list)
        if num_locs > 0:
            return alldoc_he_locs
        entity_list = []
        for entity in self.entity_set:
            if entity[-2:] == 'es':
                entity_list.append(entity[:-2])
        num_locs, alldoc_he_locs = self.find_entity_in_all_docs(entity_list)
        if num_locs > 0:
            return alldoc_he_locs
        return None

    def check_candidate_validity(self, cand: str) -> bool:
        # if length is 1 character and not a digit
        if len(cand) == 1 and ord(cand) in range(91, 123):
            return False
        if cand in ["of", "he", "a", "an", "the", "as", "e .", "s .", "a .", '*', ',', '.', '"']:
            return False
        if cand == ' '.join(self.question):
            return False
        return True

    def check_sentdist(self, entlocs: List[int],
                       candlocs: List[int], k=2) -> bool:
        """
        check if there any combination which
        falls under a specified distance threshold
        :param entlocs:
        :param candlocs:
        :param k:
        :return:
        """
        for el in entlocs:
            for cl in candlocs:
                if abs(cl - el) < self.sentlimit + k:  # window size 5
                    return True
        return False

    def find_path_for_cand(self, allents: List[Tuple[int, str]],
                           cand: str,
                           candpos: List[str], candner: List[str],
                           curidx: int, head_ent_locs: Any,
                           cand_find_window=2) -> List[Dict]:
        """
        find paths for a candidate
        :param allents:
        :param curidx:
        :param cand:
        :param candpos:
        :param candner:
        :param curidx:
        :param head_ent_locs:
        :param cand_find_window:
        :return: List[ent, docidx, List[wordidx]]
        """
        idxs = []

        # code for single hop path
        cand_locs = get_locations_words(sum(self.docsentlist[curidx], []),
                                        get_lookup_words(cand.split(' '),
                                                         candpos, candner,
                                                         type=self.candenttype),
                                        self.docoffsets[curidx],
                                        allow_partial=False)
        if len(cand_locs) > 0:
            idxs.append({"head_ent_docidx": curidx,
                         "head_ent": head_ent_locs,
                         "e1_with_head_ent": None,
                         "e1_with_head_widx": None,
                         "e1": None,
                         "e1_docidx": None,
                         "e1_locs": None,
                         "cand_locs": cand_locs})

        for entwidx, ent in allents:
            if ent == cand:
                continue
            for i in list(range(curidx)) + list(range(curidx + 1, len(self.docsentlist))):
                doctoks = sum(self.docsentlist[i], [])
                assert len(doctoks) == len(self.docoffsets[i])
                cand_locs = get_locations_words(doctoks, get_lookup_words(cand.split(' '),
                                                                          candpos, candner,
                                                                          type=self.candenttype),
                                                self.docoffsets[i],
                                                allow_partial=True)
                if len(cand_locs) > 0:
                    ent_locs = get_locations(doctoks, ent, self.docoffsets[i],
                                             allow_partial=True)
                    if len(ent_locs) > 0:
                        for eloc in ent_locs.keys():
                            idxs.append({"head_ent_docidx": curidx,
                                         "head_ent": head_ent_locs,
                                         "e1_with_head_ent": ent,
                                         "e1_with_head_widx": entwidx,
                                         "e1": eloc,
                                         "e1_docidx": i,
                                         "e1_locs": ent_locs[eloc],
                                         "cand_locs": cand_locs})
        return idxs

    def accum_paths_for_cand(self, alldoc_he_locs: List[List[Any]],
                             docners: List[List[str]],
                             docpostags: List[List[str]],
                             cand: str, candpos: List[str],
                             candner: List[str]) -> List[Any]:
        """
        accumulate all the paths for a particular candidate
        :param alldoc_he_locs:
        :param docners:
        :param docpostags:
        :param cand:
        :param candpos:
        :param candner:
        :return:
        """
        paths_to_cand = []

        # exclude the ill-posed candidates
        if not self.check_candidate_validity(cand):
            return paths_to_cand

        for docidx, doc in enumerate(self.docsentlist):
            for sentidx, sent in enumerate(doc):
                prev_num_words = len(sum(doc[:sentidx], []))
                valid_sents = doc[sentidx:min(len(doc),
                                              sentidx + self.sentlimit + 1)]
                nerstidx = prev_num_words
                nerendidx = nerstidx + len(sum(valid_sents, []))

                he_locs = deepcopy(alldoc_he_locs[docidx][sentidx])
                if len(he_locs) > 0:
                    for key in he_locs.keys():
                        for i in range(len(he_locs[key])):
                            he_locs[key][i] += prev_num_words

                    ners = docners[docidx][nerstidx:nerendidx]
                    postags = docpostags[docidx][nerstidx:nerendidx]
                    word_toks = sum(valid_sents, [])
                    assert len(ners) == len(word_toks)
                    assert len(ners) == len(postags)

                    neidxs, cwn, cne = cluster_nertags(word_toks, ners)
                    poidxs, cwp, cpo = cluster_postags(word_toks, postags)
                    neidxs = [n + nerstidx for n in neidxs]
                    poidxs = [p + nerstidx for p in poidxs]

                    entset = get_all_ents(cwn, cne, cwp, cpo, neidxs, poidxs)
                    # remove the head entity(s) from the entset
                    # entset = [es for es in entset if es[1].lower() != self.entity.lower()]
                    entset = remove_overlapping_ents(he_locs, entset)

                    paths_to_cand += self.find_path_for_cand(entset, cand, candpos, candner,
                                                             docidx, he_locs)
        return paths_to_cand

    def get_paths(self, docners: List[List[str]],
                  docpostags: List[List[str]]) -> Dict:
        """
        get path lists for all candidates
        :param docners:
        :param docpostags:
        :return:
        """
        alldoc_he_locs = None
        if len(self.question) > 0:
            alldoc_he_locs = self.get_all_he_locs()

        all_paths = {}
        for candidx, cand in enumerate(self.candidates):
            if len(self.question) == 0 or alldoc_he_locs is None:
                paths_to_cand = []
            else:
                candpos = self.candpos[candidx]
                candner = self.candner[candidx]
                paths_to_cand = self.accum_paths_for_cand(alldoc_he_locs,
                                                          docners,
                                                          docpostags,
                                                          cand, candpos, candner)

            all_paths[cand] = paths_to_cand

        return all_paths
