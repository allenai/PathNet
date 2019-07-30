from typing import Dict, Optional, AnyStr, List, Any
import time
import datetime
import logging

import torch
import gc
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, DataIterator
from allennlp.data.iterators import BucketIterator
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TimeDistributed, Highway
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training import Trainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import masked_softmax, masked_log_softmax, masked_max, masked_mean
from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.modules.span_extractors.endpoint_span_extractor import SpanExtractor

from pathnet.nn.layers import JointEncoder, AttnPooling
from pathnet.nn.util import seq2vec_seq_aggregate, pad_packed_loc_tensors, \
    pad_packed_loc_tensors_with_docidxs, gather_vectors_using_index

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("pathnet_semi_modular")
class QAMultiChoicePDCDPRAM(Model):
    """
    This ``Model`` implements the PathNet model
    This implementation is semi modular. Mainly provided for reproducing
    the reported results in the paper. For detailed documentation, follow the
    code in ``pathnet_full_modular.py``
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 aggregate_feedforward: FeedForward,
                 span_extractor: Optional[SpanExtractor],
                 he_e1wh_projector: FeedForward,
                 e1_ca_projector: FeedForward,
                 path_projector: FeedForward,
                 allchoice_projector: FeedForward,
                 question_projector: FeedForward,
                 combined_q_projector: FeedForward,
                 combined_s_projector: FeedForward,
                 joint_encoder: JointEncoder,
                 doc_aggregator: AttnPooling,
                 choice_aggregator: AttnPooling,
                 path_aggregator: FeedForward,
                 path_loc_aggregator: str = 'max',
                 question_encoder: Optional[Seq2SeqEncoder] = None,
                 document_encoder: Optional[Seq2SeqEncoder] = None,
                 choice_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 embeddings_dropout_value: Optional[float] = 0.0,
                 allchoice_loc: bool = True,
                 path_enc: bool = True,
                 path_enc_doc_based: bool = True,
                 path_enc_loc_based: bool = True,
                 combine_scores: str = 'add_cat',
                 # share_encoders: Optional[bool] = False
                 ) -> None:
        super(QAMultiChoicePDCDPRAM, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder

        if embeddings_dropout_value > 0.0:
            self._embeddings_dropout = torch.nn.Dropout(p=embeddings_dropout_value)
        else:
            self._embeddings_dropout = lambda x: x

        self._question_encoder = question_encoder  # bidirectional RNN
        self._document_encoder = document_encoder
        self._choice_encoder = choice_encoder
        self._span_extractor = span_extractor

        self._allchoice_loc = allchoice_loc
        self._path_enc = path_enc
        self._path_enc_doc_based = path_enc_doc_based
        self._path_enc_loc_based = path_enc_loc_based

        if not self._allchoice_loc and not self._path_enc:
            raise ConfigurationError("One of Allchoice Location based or "
                                     "Path encoding must have to be set True!")

        if not self._path_enc:
            self._path_enc_loc_based = False
            self._path_enc_doc_based = False
        if self._path_enc:
            if not self._path_enc_doc_based and not self._path_enc_loc_based:
                raise ConfigurationError("One of the path encoding component has to be True!")

        self._he_e1wh_projector = he_e1wh_projector
        self._e1_ca_projector = e1_ca_projector
        self._path_projector = path_projector
        self._allchoice_projector = allchoice_projector
        self._question_projector = question_projector
        self._combined_q_projector = combined_q_projector
        self._combined_s_projector = combined_s_projector
        self._aggregate_feedforward = aggregate_feedforward
        self._path_loc_aggregator = path_loc_aggregator
        self._choice_aggregator = choice_aggregator
        self._joint_encoder = joint_encoder
        self._doc_aggregator = doc_aggregator
        self._path_aggregator = path_aggregator

        if self._path_loc_aggregator == 'max':
            self._agg_func = masked_max
        elif self._path_loc_aggregator == 'mean':
            self._agg_func = masked_mean
        else:
            raise NotImplementedError
        self._combine_scores = combine_scores

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.NLLLoss()

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                documents: Dict[str, torch.LongTensor],
                candidates: Dict[str, torch.LongTensor],
                flattened_p1list: torch.LongTensor,
                flattened_p1list_e1wh: torch.LongTensor,
                flattened_p2list_e1: torch.LongTensor,
                flattened_p2list: torch.LongTensor,
                flat_he_spans: torch.LongTensor,
                flat_e1wh_spans: torch.LongTensor,
                flat_e1_spans: torch.LongTensor,
                flat_choice_spans: torch.LongTensor,
                all_choice_docidxs: torch.LongTensor,
                all_choice_locs: torch.LongTensor,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """

        :param question:
        :param documents:
        :param candidates:
        :param flattened_p1list:
        :param flattened_p1list_e1wh:
        :param flattened_p2list_e1:
        :param flattened_p2list:
        :param flat_he_spans:
        :param flat_e1wh_spans:
        :param flat_e1_spans:
        :param flat_choice_spans:
        :param all_choice_docidxs:
        :param all_choice_locs:
        :param label:
        :param metadata:
        :return:
        Returns
        -------
        An output dictionary consisting of:
        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # Embedding
        # ----------------------------------------
        embedded_question = self._text_field_embedder(question)
        embedded_question = self._embeddings_dropout(embedded_question)  # B * T1 * d
        embedded_documents = self._text_field_embedder(documents)
        embedded_documents = self._embeddings_dropout(embedded_documents)  # B * N * T2 * d
        embedded_choices = self._text_field_embedder(candidates)  # B * C * T3 * d
        embedded_choices = self._embeddings_dropout(embedded_choices)
        # masks
        # ----------------------------------------
        question_mask = get_text_field_mask(question).float()  # B * T1
        documents_mask = get_text_field_mask(documents, num_wrapping_dims=1).float()  # B * N * T2
        candidate_mask = get_text_field_mask(candidates, num_wrapping_dims=1).float()  # B * C * T3

        if self._path_enc:
            flat_he_spans_mask = (flat_he_spans != -1).long()[:, :, 0]  # B * CPL
            flat_e1wh_spans_mask = (flat_e1wh_spans != -1).long()[:, :, 0]
            flat_e1_spans_mask = (flat_e1_spans != -1).long()[:, :, 0]
            flat_choice_spans_mask = (flat_choice_spans != -1).long()[:, :, 0]
        if self._allchoice_loc:
            allchoice_locs_mask = (all_choice_locs != -1).long()  # B * C * L * 2
            allchoice_locs_mask = allchoice_locs_mask[:, :, :, 0]  # B * C * L

        if self._path_enc:
            flattened_p1list = flattened_p1list * flat_he_spans_mask.unsqueeze(-1)  # B * CPL * 1
            flattened_p1list_e1wh = flattened_p1list_e1wh * flat_e1wh_spans_mask.unsqueeze(-1)  # B * CPL * 1
            flattened_p2list_e1 = flattened_p2list_e1 * flat_e1_spans_mask.unsqueeze(-1)  # B * CPL * 1
            flattened_p2list = flattened_p2list * flat_choice_spans_mask.unsqueeze(-1)  # B * CPL * 1

        # tensor sizes
        # ----------------------------------------
        batch_size = embedded_documents.shape[0]  # B
        num_docs = embedded_documents.shape[1]
        num_doc_tokens = embedded_documents.shape[2]
        choices_cnt = embedded_choices.shape[1]  # C
        num_choice_tokens = embedded_choices.shape[2]
        batch_size_range = range(batch_size)
        if self._path_enc:
            paths_cnt = max([metadata[i]['max_num_paths'] for i in batch_size_range])  # P
            num_locs_he = max([metadata[i]['max_num_he_locs'] for i in batch_size_range])
            num_locs_e1wh = max([metadata[i]['max_num_e1wh_locs'] for i in batch_size_range])
            num_locs_e1 = max([metadata[i]['max_num_e1_locs'] for i in batch_size_range])
            num_locs_choice = max([metadata[i]['max_num_ca_locs'] for i in batch_size_range])
        if self._allchoice_loc:
            num_locs_allchoice = all_choice_locs.shape[2]

        # Encoding
        # ----------------------------------------
        if self._question_encoder:
            embedded_question = self._question_encoder(embedded_question, question_mask)  # B * T * H
            embedded_question = self._embeddings_dropout(embedded_question)

        if self._choice_encoder:
            embedded_choices = self._choice_encoder(embedded_choices.view(batch_size * choices_cnt,
                                                                          num_choice_tokens, -1),
                                                    candidate_mask.view(batch_size * choices_cnt,
                                                                        -1))  # BC * T3 * H
            embedded_choices = self._embeddings_dropout(embedded_choices)
        else:
            embedded_choices = embedded_choices.view(batch_size * choices_cnt, num_choice_tokens, -1)

        embedded_choices = self._choice_aggregator(embedded_choices,
                                                   candidate_mask.view(batch_size * choices_cnt,
                                                                       -1))  # BC * H
        embedded_choices = embedded_choices.view(batch_size, choices_cnt, -1)  # B * C * H

        if self._document_encoder:
            embedded_documents = self._document_encoder(embedded_documents.view(batch_size * num_docs,
                                                                                num_doc_tokens, -1),
                                                        documents_mask.view(batch_size * num_docs,
                                                                            -1))  # BN * T2 * H
            embedded_documents = embedded_documents.view(batch_size, num_docs * num_doc_tokens,
                                                         -1)  # B * NT2 * H
            embedded_documents = self._embeddings_dropout(embedded_documents)
        else:
            embedded_documents = embedded_documents.view(batch_size, num_docs * num_doc_tokens, -1)
        documents_mask = documents_mask.view(batch_size, -1)  # B * NT2

        if self._path_enc:
            flat_he_spans = flat_he_spans + flattened_p1list * num_doc_tokens
            flat_e1wh_spans = flat_e1wh_spans + flattened_p1list_e1wh * num_doc_tokens
            flat_e1_spans = flat_e1_spans + flattened_p2list_e1 * num_doc_tokens
            flat_choice_spans = flat_choice_spans + flattened_p2list * num_doc_tokens

            extracted_he = self._span_extractor(embedded_documents, flat_he_spans,
                                                documents_mask, flat_he_spans_mask)  # B * cpl * 2H
            extracted_e1wh = self._span_extractor(embedded_documents, flat_e1wh_spans,
                                                  documents_mask, flat_e1wh_spans_mask)  # B * cpl * 2H
            extracted_e1 = self._span_extractor(embedded_documents, flat_e1_spans,
                                                documents_mask, flat_e1_spans_mask)  # B * cpl * 2H
            extracted_choice = self._span_extractor(embedded_documents, flat_choice_spans,
                                                    documents_mask, flat_choice_spans_mask)  # B * cpl * 2H

            he_track_list = [metadata[i]['he_tracks'] for i in batch_size_range]
            e1wh_track_list = [metadata[i]['e1wh_tracks'] for i in batch_size_range]
            e1_track_list = [metadata[i]['e1_tracks'] for i in batch_size_range]
            choice_track_list = [metadata[i]['choice_tracks'] for i in batch_size_range]
            extracted_he, p1list, he_spans_mask = \
                pad_packed_loc_tensors_with_docidxs(extracted_he,
                                                    flattened_p1list,
                                                    choices_cnt, paths_cnt,
                                                    num_locs_he, he_track_list,
                                                    flat_he_spans_mask)  # B * C * P * L * 2H
            extracted_e1wh, e1wh_spans_mask = pad_packed_loc_tensors(extracted_e1wh, choices_cnt, paths_cnt,
                                                                     num_locs_e1wh, e1wh_track_list,
                                                                     flat_e1wh_spans_mask)
            extracted_e1, e1_spans_mask = pad_packed_loc_tensors(extracted_e1, choices_cnt, paths_cnt,
                                                                 num_locs_e1, e1_track_list,
                                                                 flat_e1_spans_mask)
            extracted_choice, p2list, choice_spans_mask = \
                pad_packed_loc_tensors_with_docidxs(extracted_choice,
                                                    flattened_p2list,
                                                    choices_cnt,
                                                    paths_cnt,
                                                    num_locs_choice, choice_track_list,
                                                    flat_choice_spans_mask)

            flat_he_spans_mask = he_spans_mask.view(-1, num_locs_he)  # BCP * L
            flat_e1wh_spans_mask = e1wh_spans_mask.view(-1, num_locs_e1wh)
            flat_e1_spans_mask = e1_spans_mask.view(-1, num_locs_e1)
            flat_choice_spans_mask = choice_spans_mask.view(-1, num_locs_choice)

            if self._path_enc_doc_based:
                p1list = p1list[:, :, :, 0]  # B * C * P * 1
                p2list = p2list[:, :, :, 0]  # B * C * P * 1
                # qdep doc_encoding
                joint_encoding = self._joint_encoder(embedded_documents.view(batch_size,
                                                                             num_docs,
                                                                             num_doc_tokens, -1),
                                                     embedded_question, documents_mask.view(batch_size,
                                                                                            num_docs, -1),
                                                     question_mask)  # B * N * T2 * 2H
                joint_encoding = self._embeddings_dropout(joint_encoding)
                joint_encoding = self._doc_aggregator(joint_encoding.view(batch_size * num_docs,
                                                                          num_doc_tokens, -1),
                                                      documents_mask.view(batch_size * num_docs, -1))  # BN * 2H
                joint_encoding = joint_encoding.view(batch_size, num_docs, -1)  # B * N * 2H

            # location-based extraction
            if self._path_enc_loc_based:
                extracted_he = extracted_he.view(batch_size * choices_cnt * paths_cnt,
                                                 num_locs_he, -1)  # BCP * L * 2H
                extracted_e1wh = extracted_e1wh.view(batch_size * choices_cnt * paths_cnt, num_locs_e1wh, -1)
                extracted_e1 = extracted_e1.view(batch_size * choices_cnt * paths_cnt, num_locs_e1, -1)
                extracted_choice = extracted_choice.view(batch_size * choices_cnt * paths_cnt,
                                                         num_locs_choice, -1)

                extracted_he = extracted_he * flat_he_spans_mask.unsqueeze(-1).float()
                extracted_e1wh = extracted_e1wh * flat_e1wh_spans_mask.unsqueeze(-1).float()
                extracted_e1 = extracted_e1 * flat_e1_spans_mask.unsqueeze(-1).float()
                extracted_choice = extracted_choice * flat_choice_spans_mask.unsqueeze(-1).float()

                extracted_he = self._agg_func(extracted_he, flat_he_spans_mask.unsqueeze(-1),
                                              dim=1, keepdim=False)  # BCP * 2H
                extracted_e1wh = self._agg_func(extracted_e1wh, flat_e1wh_spans_mask.unsqueeze(-1),
                                                dim=1, keepdim=False)
                extracted_e1 = self._agg_func(extracted_e1, flat_e1_spans_mask.unsqueeze(-1),
                                              dim=1, keepdim=False)
                extracted_choice = self._agg_func(extracted_choice, flat_choice_spans_mask.unsqueeze(-1),
                                                  dim=1, keepdim=False)

        if self._allchoice_loc:
            # extraction of the representations of candidate appearances
            all_choice_locs = all_choice_locs + all_choice_docidxs * num_doc_tokens
            flat_all_choice_locs = all_choice_locs.view(batch_size,
                                                        choices_cnt * num_locs_allchoice, -1)  # B * CL * 2
            flat_allchoice_locs_mask = allchoice_locs_mask.view(batch_size, -1)  # B * CL
            extracted_allchoice_locs = self._span_extractor(embedded_documents, flat_all_choice_locs,
                                                            documents_mask,
                                                            flat_allchoice_locs_mask)  # B * CL * 2H
            extracted_allchoice_locs = self._allchoice_projector(extracted_allchoice_locs)  # B * CL * H
            extracted_allchoice_locs = extracted_allchoice_locs.view(batch_size * choices_cnt,
                                                                     num_locs_allchoice, -1)  # BC * L * H

        # path encoding
        # ----------------------------------------
        if self._path_enc:
            he_mask_ = (torch.sum(flat_he_spans_mask, -1) > 0).unsqueeze(1).long()
            choice_mask_ = (torch.sum(flat_choice_spans_mask, -1) > 0).unsqueeze(1).long()  # BCP * 1
            if self._path_enc_loc_based:
                he_e1wh_rep = self._he_e1wh_projector(torch.cat([extracted_he, extracted_e1wh], 1))
                he_e1wh_rep = he_e1wh_rep * he_mask_.float()
                e1_ca_rep = self._e1_ca_projector(torch.cat([extracted_e1, extracted_choice], 1))
                e1_ca_rep = e1_ca_rep * choice_mask_.float()
                encoded_paths = self._path_projector(torch.cat([he_e1wh_rep, e1_ca_rep], 1))
                encoded_paths = encoded_paths * choice_mask_.float()
                encoded_paths = encoded_paths.view(batch_size * choices_cnt, paths_cnt, -1)  # BC * P * H

            if self._path_enc_doc_based:
                extracted_p1 = gather_vectors_using_index(joint_encoding, p1list)  # B * C * P * 2H
                extracted_p2 = gather_vectors_using_index(joint_encoding, p2list)  # B * C * P * 2H
                extracted_p1 = extracted_p1 * he_mask_.view(batch_size, choices_cnt, paths_cnt, 1).float()
                extracted_p2 = extracted_p2 * choice_mask_.view(batch_size, choices_cnt, paths_cnt, 1).float()
                combined_p1p2 = self._path_aggregator(torch.cat([extracted_p1,
                                                                 extracted_p2], -1))  # B * C * P * H
                raw_scores = combined_p1p2.view(batch_size * choices_cnt,
                                                paths_cnt, -1).bmm(embedded_choices.view(batch_size * choices_cnt,
                                                                                         -1, 1))  # BC * P * 1
                raw_scores = raw_scores.view(batch_size, choices_cnt, paths_cnt)  # B * C * P

        # question formulation
        # ----------------------------------------
        q1st = embedded_question[:, 0, :]  # B * H
        qlast = seq2vec_seq_aggregate(embedded_question, question_mask, aggregate='last',
                                      bidirectional=False, dim=1)  # B * H
        qh = torch.cat([q1st, qlast], -1)  # B * 2H
        qh = self._question_projector(qh)  # B * H

        # attention calculation
        # ----------------------------------------
        attn_q_projected = self._combined_q_projector(qh)  # B * H
        expand_qh = attn_q_projected.unsqueeze(1)  # B * 1 * H

        if self._allchoice_loc:
            allchoice_locs_projected = self._combined_s_projector(extracted_allchoice_locs)  # BC * L * H
            allchoice_locs_projected = allchoice_locs_projected.view(batch_size,
                                                                     choices_cnt * num_locs_allchoice,
                                                                     -1)  # B * CL * H
            combined_allc_locs_projected = torch.tanh(expand_qh + allchoice_locs_projected)  # B * CL * H
            attn_allc_locs = self._aggregate_feedforward(combined_allc_locs_projected)  # B * CL * 1
            attn_allc_locs = attn_allc_locs.squeeze(-1).view(batch_size, choices_cnt,
                                                             num_locs_allchoice)  # B * C * L
            allc_mask = allchoice_locs_mask.view(batch_size, choices_cnt, -1)  # B * C * L

        if self._path_enc and self._path_enc_loc_based:
            attn_path_locs_projected = self._combined_s_projector(encoded_paths)  # BC * P * H
            attn_path_locs_projected = attn_path_locs_projected.view(batch_size,
                                                                     choices_cnt * paths_cnt,
                                                                     -1)  # B * CP * H
            combined_path_locs_projected = torch.tanh(expand_qh + attn_path_locs_projected)  # B * CP * H
            attn_path_locs = self._aggregate_feedforward(combined_path_locs_projected)  # B * CP * 1
            attn_path_locs = attn_path_locs.squeeze(-1).view(batch_size, choices_cnt, paths_cnt)  # B * C * P

        if self._path_enc:
            path_mask = choice_mask_.squeeze(-1).view(batch_size, choices_cnt, -1)  # B * C * P

        if self._path_enc_loc_based and self._path_enc_doc_based and self._allchoice_loc:
            if self._combine_scores == 'cat_all':
                attn_forward = torch.cat([raw_scores, attn_path_locs, attn_allc_locs], 2)  # B * C * (2P+L)
                attn_mask = torch.cat([path_mask, path_mask, allc_mask], 2)  # B * C * 2P+L
            elif self._combine_scores == "add_cat":
                attn_forward = torch.cat([raw_scores + attn_path_locs, attn_allc_locs], 2)  # B * C * (P+L)
                attn_mask = torch.cat([path_mask, allc_mask], 2)  # B * C * (P+L)
            else:
                NotImplementedError
        elif self._path_enc_loc_based and self._allchoice_loc:
            attn_forward = torch.cat([attn_path_locs, attn_allc_locs], 2)  # B * C * (P+L)
            attn_mask = torch.cat([path_mask, allc_mask], 2)  # B * C * (P+L)
        elif self._path_enc_doc_based and self._allchoice_loc:
            attn_forward = torch.cat([raw_scores, attn_allc_locs], 2)  # B * C * (P+L)
            attn_mask = torch.cat([path_mask, allc_mask], 2)  # B * C * (P+L)
        elif self._path_enc and not self._allchoice_loc:
            if self._path_enc_doc_based and self._path_enc_loc_based:
                attn_forward = raw_scores + attn_path_locs  # B * C * P
                attn_mask = path_mask  # B * C * P
            elif self._path_enc_loc_based:
                attn_forward = attn_path_locs  # B * C * P
                attn_mask = path_mask  # B * C * P
            elif self._path_enc_doc_based:
                attn_forward = raw_scores  # B * C * P
                attn_mask = path_mask  # B * C * P
            else:
                raise NotImplementedError
        elif not self._path_enc and self._allchoice_loc:
            attn_forward = attn_allc_locs  # B * C * L
            attn_mask = allc_mask  # B * C * L
        else:
            raise NotImplementedError

        attn_forward = attn_forward.view(batch_size, -1)  # B * C(2P+L) / B * C(P+L) / B * CP / B * CL
        attn_mask = attn_mask.view(batch_size, -1)  # B * C(2P+L) / B * C(P+L)

        normalized_attn = masked_softmax(attn_forward, attn_mask, dim=-1)
        normalized_attn = normalized_attn.view(batch_size, choices_cnt, -1)
        output = torch.sum(normalized_attn, -1)  # B * C

        label_logits = (output + 1e-45).log()

        output_dict = {
            "label_logits": label_logits,
            "label_probs": output,
            "path_probs": normalized_attn,
            "metadata": metadata}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }
