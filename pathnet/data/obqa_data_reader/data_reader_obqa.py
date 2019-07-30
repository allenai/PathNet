from typing import Dict, List, Any, Tuple
import json
import logging

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ListField, MetadataField, \
    IndexField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from allennlp.data.tokenizers import word_splitter

from pathnet.data.data_utils import pack_span_idxs, find_cand_locs_fromdoclist, \
    get_locs_forall, get_max_locs, pack_doc_idxs

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("obqa_data_reader")
class OBQAMultiChoiceJsonReader(DatasetReader):
    """
    This data is formatted as jsonl, one json-formatted instance per line.
    This dataset format is obtained after the path adjustment step.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for all.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        See :class:`TokenIndexer`.
    lazy : whether to use lazy mode for training (do not keep everything in the memory)
    cut_context : Whether to strip out the portions from documents which do not
        participate in constructing paths
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 cut_context: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(
            word_splitter=word_splitter.JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._cut_context = cut_context

    @staticmethod
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

    def get_documents_markers(self, docsents: List[List[List[str]]],
                              paths: List[List[Any]]) -> Any:
        """
        Cut the documents
        :param docsents:
        :param paths:
        :return:
        """
        if not self._cut_context:
            return None
        min_max_sentidx_dict = {}
        num_word_shift_dict = {}
        for didx in range(len(docsents)):
            min_max_sentidx_dict[didx] = [len(docsents[didx]) - 1, 0]
            num_word_shift_dict[didx] = 0

        for cidx in range(len(paths)):
            for pidx in range(len(paths[cidx])):
                he_docidx = paths[cidx][pidx]['he_docidx']
                ca_docidx = paths[cidx][pidx]['cand_docidx']
                he_locs = paths[cidx][pidx]['he_locs']
                e1wh_locs = paths[cidx][pidx]['e1wh_loc']
                e1_locs = paths[cidx][pidx]['e1_locs']
                ca_locs = paths[cidx][pidx]['cand_locs']

                # if he_docidx is not None and ca_docidx is not None
                if he_docidx == ca_docidx:
                    all_start_sent_inds = []
                    all_end_sent_inds = []
                    for x in [he_locs, ca_locs]:
                        if x is not None:
                            for loc in x:
                                if loc[0] != -1 or loc[1] != -1:
                                    all_start_sent_inds.append(self.find_sentidx(
                                        docsents[he_docidx], loc[0])
                                    )
                                    all_end_sent_inds.append(self.find_sentidx(
                                        docsents[he_docidx], loc[1])
                                    )
                    if len(all_start_sent_inds) > 0:
                        min_max_sentidx_dict[he_docidx][0] = min(min(all_start_sent_inds),
                                                                 min_max_sentidx_dict[he_docidx][0])
                    if len(all_end_sent_inds) > 0:
                        min_max_sentidx_dict[he_docidx][1] = max(max(all_end_sent_inds),
                                                                 min_max_sentidx_dict[he_docidx][1])
                else:
                    idxs = [he_docidx, ca_docidx]
                    for yidx, y in enumerate([[he_locs, e1wh_locs], [e1_locs, ca_locs]]):
                        docidx = idxs[yidx]
                        all_start_sent_inds = []
                        all_end_sent_inds = []
                        for x in y:
                            if x is not None:
                                for loc in x:
                                    if loc[0] != -1 or loc[1] != -1:
                                        all_start_sent_inds.append(self.find_sentidx(
                                            docsents[docidx], loc[0])
                                        )
                                        all_end_sent_inds.append(self.find_sentidx(
                                            docsents[docidx], loc[1])
                                        )
                        if len(all_start_sent_inds) > 0:
                            min_max_sentidx_dict[docidx][0] = min(min(all_start_sent_inds),
                                                                  min_max_sentidx_dict[docidx][0])
                        if len(all_end_sent_inds) > 0:
                            min_max_sentidx_dict[docidx][1] = max(max(all_end_sent_inds),
                                                                  min_max_sentidx_dict[docidx][1])

        for key in min_max_sentidx_dict.keys():
            if min_max_sentidx_dict[key][0] > min_max_sentidx_dict[key][1]:
                min_max_sentidx_dict[key] = [0, 0]  # useless doc
            if min_max_sentidx_dict[key][0] > 0:
                num_word_shift_dict[key] = len(sum(docsents[key][:min_max_sentidx_dict[key][0]], []))

        return min_max_sentidx_dict, num_word_shift_dict

    @overrides
    def _read(self, file_path: str):

        with open(file_path, 'r') as data_file:
            logger.info("Reading OBQA instances from jsonl data at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())

                item_id = item_json["id"]
                question = item_json['question']
                docsents = item_json['docsents']
                candidates_ = item_json['candidates']
                orig_paths = item_json['paths']  # cands * num_paths
                if len(candidates_) != len(orig_paths):
                    logger.info(f"******** {item_id} skipped ********")
                    continue
                ans = ' '.join(item_json["answer"]) if "answer" in item_json else None

                yield self.text_to_instance(item_id, docsents,
                                            question, candidates_, orig_paths,
                                            ans)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id, docsents,
                         question, candidates_, orig_paths,
                         ans):
        # filtering
        orig_candidates = [' '.join(c) for c in candidates_]
        assert len(orig_candidates) == len(orig_paths)
        candidates = []
        paths = []
        for c, p in zip(orig_candidates, orig_paths):
            if len(c) == 1 and ord(c) in range(91, 123):
                continue
            elif c in ["of", "he", "a", "an", "the", "as",
                       "e .", "s .", "a .", '*', ',', '.', '"']:
                continue
            else:
                candidates.append(c)
                paths.append(p)

        choice_label_to_id: Dict = {}
        choice_text_list: List[str] = []

        for choice_id, choice_text in enumerate(candidates):
            choice_label_to_id[choice_text] = choice_id
            choice_text_list.append(choice_text)

        if ans is not None:
            answer_id = choice_label_to_id[ans]
        else:
            answer_id = None

        question_text = " ".join(question)

        if self._cut_context and \
                self.get_documents_markers(docsents, paths) is not None:
            doc_sent_boundary_dict, num_word_shift_dict = self.get_documents_markers(
                docsents, paths)
            mod_docsents = []
            for didx in range(len(docsents)):
                start_sidx = doc_sent_boundary_dict[didx][0]
                end_sidx = doc_sent_boundary_dict[didx][1]
                mod_docsents.append(docsents[didx][start_sidx:end_sidx + 1])
        else:
            mod_docsents = docsents
            doc_sent_boundary_dict, num_word_shift_dict = None, None

        documents_text_list: List[str] = [' '.join(sum(doc, []))
                                          for doc in mod_docsents]

        p1_list, p2_list, he_locs_list, e1wh_locs_list, \
            e1_locs_list, ca_locs_list = get_locs_forall(paths,
                                                         doc_sent_boundary_dict,
                                                         num_word_shift_dict,
                                                         mod_docsents)

        max_he_locs, max_e1wh_locs, max_e1_locs, max_ca_locs = get_max_locs(paths,
                                                                            he_locs_list,
                                                                            e1wh_locs_list,
                                                                            e1_locs_list,
                                                                            ca_locs_list)

        flattened_he_locs_list, he_tracks = pack_span_idxs(he_locs_list)
        flattened_e1wh_locs_list, e1wh_tracks = pack_span_idxs(e1wh_locs_list)
        flattened_e1_locs_list, e1_tracks = pack_span_idxs(e1_locs_list)
        flattened_ca_locs_list, ca_tracks = pack_span_idxs(ca_locs_list)

        flattened_p1_list = pack_doc_idxs(p1_list, he_locs_list)
        flattened_p1_list_e1wh = pack_doc_idxs(p1_list, e1wh_locs_list)
        flattened_p2_list_e1 = pack_doc_idxs(p2_list, e1_locs_list)
        flattened_p2_list = pack_doc_idxs(p2_list, ca_locs_list)

        max_paths = max([len(p) for p in paths])

        all_choice_locs, all_choice_docidxs = find_cand_locs_fromdoclist(documents_text_list,
                                                                         choice_text_list,
                                                                         lowercase=True)

        return self.formatted_text_to_instance(item_id, question_text,
                                               documents_text_list,
                                               flattened_p1_list,
                                               flattened_p1_list_e1wh,
                                               flattened_p2_list_e1,
                                               flattened_p2_list,
                                               flattened_he_locs_list,
                                               flattened_e1wh_locs_list,
                                               flattened_e1_locs_list,
                                               flattened_ca_locs_list,
                                               he_tracks, e1wh_tracks,
                                               e1_tracks, ca_tracks,
                                               max_paths,
                                               max_he_locs, max_e1wh_locs,
                                               max_e1_locs, max_ca_locs,
                                               choice_text_list,
                                               all_choice_locs, all_choice_docidxs,
                                               answer_id)

    def formatted_text_to_instance(self,  # type: ignore
                                   item_id: Any,
                                   question_text: str,
                                   documents_text_list: List[str],
                                   flattened_p1_list: List[int],
                                   flattened_p1_list_e1wh: List[int],
                                   flattened_p2_list_e1: List[int],
                                   flattened_p2_list: List[int],
                                   flattened_he_locs_list: List[Tuple[int, int]],
                                   flattened_e1wh_locs_list: List[Tuple[int, int]],
                                   flattened_e1_locs_list: List[Tuple[int, int]],
                                   flattened_ca_locs_list: List[Tuple[int, int]],
                                   he_tracks: List[List[int]],
                                   e1wh_tracks: List[List[int]],
                                   e1_tracks: List[List[int]],
                                   ca_tracks: List[List[int]],
                                   max_paths: int,
                                   max_he_locs: int, max_e1wh_locs: int,
                                   max_e1_locs: int, max_ca_locs: int,
                                   choice_text_list: List[str],
                                   all_choice_locs: List[List[Tuple[int, int]]],
                                   all_choice_docidxs: List[List[int]],
                                   answer_id: int) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question_text)
        documents_list_tokens = [self._tokenizer.tokenize(dt) for dt in documents_text_list]
        if len(sum(documents_list_tokens, [])) == 0:
            documents_list_tokens = [question_tokens]

        choices_list_tokens = [self._tokenizer.tokenize(x) for x in choice_text_list]

        fields['question'] = TextField(question_tokens, self._token_indexers)
        document_text_fields = [TextField(x, self._token_indexers) for x in documents_list_tokens]
        document_field = ListField(document_text_fields)
        fields['documents'] = document_field
        fields['candidates'] = ListField([TextField(x, self._token_indexers) for x in choices_list_tokens])

        fields['flattened_p1list'] = ListField([IndexField(x, document_field)
                                                for x in flattened_p1_list])
        fields['flattened_p1list_e1wh'] = ListField([IndexField(x, document_field)
                                                     for x in flattened_p1_list_e1wh])
        fields['flattened_p2list_e1'] = ListField([IndexField(x, document_field)
                                                   for x in flattened_p2_list_e1])
        fields['flattened_p2list'] = ListField([IndexField(x, document_field)
                                                for x in flattened_p2_list])

        fields['flat_he_spans'] = ListField([SpanField(x[0], x[1], document_text_fields[flattened_p1_list[xidx]])
                                             for xidx, x in enumerate(flattened_he_locs_list)])
        fields['flat_e1wh_spans'] = ListField([SpanField(x[0], x[1], document_text_fields[flattened_p1_list_e1wh[xidx]])
                                               for xidx, x in enumerate(flattened_e1wh_locs_list)])
        fields['flat_e1_spans'] = ListField([SpanField(x[0], x[1], document_text_fields[flattened_p2_list_e1[xidx]])
                                             for xidx, x in enumerate(flattened_e1_locs_list)])
        fields['flat_choice_spans'] = ListField([SpanField(x[0], x[1], document_text_fields[flattened_p2_list[xidx]])
                                                 for xidx, x in enumerate(flattened_ca_locs_list)])

        # all choice fields
        all_choice_docidx_field = []
        all_choice_span_fileds = []
        for choice_docidxs, choice_spans in zip(all_choice_docidxs, all_choice_locs):
            all_choice_docidx_field.append(ListField([IndexField(x, document_field)
                                                      for x in choice_docidxs]))
            all_choice_span_fileds.append(ListField([SpanField(x[0], x[1],
                                                               document_text_fields[choice_docidxs[xidx]])
                                                     for xidx, x in enumerate(choice_spans)]))
        fields['all_choice_docidxs'] = ListField(all_choice_docidx_field)
        fields['all_choice_locs'] = ListField(all_choice_span_fileds)

        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "documents_text": documents_text_list,
            "choice_text_list": choice_text_list,
            "he_tracks": he_tracks,
            "e1wh_tracks": e1wh_tracks,
            "e1_tracks": e1_tracks,
            "choice_tracks": ca_tracks,
            "max_num_paths": max_paths,
            "max_num_he_locs": max_he_locs,
            "max_num_e1wh_locs": max_e1wh_locs,
            "max_num_e1_locs": max_e1_locs,
            "max_num_ca_locs": max_ca_locs,
        }

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
