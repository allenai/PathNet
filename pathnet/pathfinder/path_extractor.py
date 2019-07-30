from typing import List, Dict, Any, Tuple
from copy import deepcopy

from pathnet.pathfinder.util import create_offsets, get_locations, \
    cluster_postags, cluster_nertags, get_all_ents, find_sentidx, remove_overlapping_ents


class PathFinder(object):
    """class for path finder """

    def __init__(self, qid, docsentlist: List[List[List[str]]],
                 entity: str, relation: str,
                 candidates: List[str], answer: str = None,
                 sentlimit: int = 1, minentlen: int = 3,
                 nearest_only: bool = False) -> None:
        self.docsentlist = docsentlist
        self.docoffsets = []
        for docsent in self.docsentlist:
            doctoks = sum(docsent, [])
            self.docoffsets.append(create_offsets(doctoks))
        self.entity = entity
        self.relation = relation
        self.candidates = candidates
        self.answer = answer
        self.sentlimit = sentlimit
        self.minentlen = minentlen
        self.nearest_only = nearest_only
        self.id = qid

    def find_entity_in_all_docs(self, entity_list: List[str]) -> (int, List[List[Any]]):
        """
        find the entities in all the documents
        :param entity_list:
        :return:
        """
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
        get the locations of the head entity
        :return:
        """
        num_locs, alldoc_he_locs = self.find_entity_in_all_docs([self.entity])
        if num_locs > 0:
            return alldoc_he_locs
        entity_list = []
        if "-" in self.entity:
            entity_list.append(self.entity.replace('-', ' - '))
        if " of " in self.entity:
            entity_list.append(self.entity.split(" of ")[0].strip())
        if ":" in self.entity:
            entity_list.append(self.entity.split(":")[0].strip())
        num_locs, alldoc_he_locs = self.find_entity_in_all_docs(entity_list)
        if num_locs > 0:
            return alldoc_he_locs
        if self.entity[-1:] == 's':
            num_locs, alldoc_he_locs = self.find_entity_in_all_docs([self.entity[:-1]])
            if num_locs > 0:
                return alldoc_he_locs
        if self.entity[-2:] == 'es':
            num_locs, alldoc_he_locs = self.find_entity_in_all_docs([self.entity[:-1]])
            if num_locs > 0:
                return alldoc_he_locs
        num_toks = len(self.entity.split())
        part1 = int(num_toks / 2)
        first_part = ' '.join(self.entity.split()[:part1 + 1])
        last_part = ' '.join(self.entity.split()[part1 + 1:])
        _, alldoc_he_locs = self.find_entity_in_all_docs([first_part, last_part])
        return alldoc_he_locs

    def check_candidate_validity(self, cand: str) -> bool:
        # if length is 1 character and not a digit
        if len(cand) == 1 and ord(cand) in range(91, 123):
            return False
        if cand in ["of", "he", "a", "an", "the", "as", "e .", "s .", "a .", '*', ',', '.', '"']:
            return False
        if cand == self.entity:
            return False
        return True

    def check_sentdist(self, entlocs: List[int],
                       candlocs: List[int], k=2) -> bool:
        """
        check if there any combination which
        falls under a specified sentence distance threshold
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
                           curidx: int, head_ent_locs: Any,
                           cand_find_window=2) -> List[Dict]:
        """
        find the paths for a particular candidate
        :param allents:
        :param curidx:
        :param cand:
        :param curidx:
        :param head_ent_locs:
        :param cand_find_window:
        :return: List[ent, docidx, List[wordidx]]
        """
        idxs = []

        # code for single hop path
        cand_locs = get_locations(sum(self.docsentlist[curidx], []),
                                  cand, self.docoffsets[curidx],
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
                cand_locs = get_locations(doctoks, cand, self.docoffsets[i],
                                          allow_partial=False)
                if len(cand_locs) > 0:
                    ent_locs = get_locations(doctoks, ent, self.docoffsets[i],
                                             allow_partial=True)
                    if len(ent_locs) > 0:
                        for eloc in ent_locs.keys():
                            ewlocs = ent_locs[eloc]
                            eslocs = [find_sentidx(self.docsentlist[i], ew) for
                                      ew in ewlocs]
                            cand_sent_locs = {}
                            for key in cand_locs.keys():
                                cand_sent_locs[key] = [find_sentidx(
                                    self.docsentlist[i], cw)
                                    for cw in cand_locs[key]]
                            cand_sent_locs_values = sum(list(cand_sent_locs.values()), [])

                            # check for atleast self.sentlimit + k sent gap
                            if self.check_sentdist(eslocs, cand_sent_locs_values,
                                                   k=cand_find_window):
                                # check the closest ones
                                valid_cand_locs = {}
                                if self.nearest_only:
                                    for window in range(cand_find_window + 1):
                                        valid_cand_locs = {}
                                        for key in cand_locs.keys():
                                            valid_cand_locs_for_key = [cand_locs[key][i]
                                                                       for i in range(len(cand_locs[key]))
                                                                       if self.check_sentdist(
                                                    eslocs, [cand_sent_locs[key][i]], k=window
                                                )]
                                            if len(valid_cand_locs_for_key) > 0:
                                                valid_cand_locs[key] = valid_cand_locs_for_key
                                        if len(sum(list(valid_cand_locs.values()), [])) > 0:
                                            break
                                else:
                                    valid_cand_locs = cand_locs
                                idxs.append({"head_ent_docidx": curidx,
                                             "head_ent": head_ent_locs,
                                             "e1_with_head_ent": ent,
                                             "e1_with_head_widx": entwidx,
                                             "e1": eloc,
                                             "e1_docidx": i,
                                             "e1_locs": ent_locs[eloc],
                                             "cand_locs": valid_cand_locs})
        return idxs

    def accum_paths_for_cand(self, alldoc_he_locs: List[List[Any]],
                             docners: List[List[str]],
                             docpostags: List[List[str]],
                             cand: str) -> List[Any]:
        """
        accumulate all the paths for a particular candidate
        :param alldoc_he_locs:
        :param docners:
        :param docpostags:
        :param cand:
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

                    paths_to_cand += self.find_path_for_cand(entset, cand,
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
        if len(self.entity.split()) > 0:
            alldoc_he_locs = self.get_all_he_locs()

        all_paths = {}
        for cand in self.candidates:
            if len(self.entity.split()) == 0:
                paths_to_cand = []
            else:
                paths_to_cand = self.accum_paths_for_cand(alldoc_he_locs,
                                                          docners,
                                                          docpostags,
                                                          cand)

            all_paths[cand] = paths_to_cand

        return all_paths
