from typing import Optional
import logging
import torch
from allennlp.common.from_params import FromParams
from allennlp.modules import FeedForward, Seq2SeqEncoder
from allennlp.nn.util import masked_softmax

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CtxPathEncoder(torch.nn.Module, FromParams):
    """
    Context-based Path Encoder Class
    """
    def __init__(self, type='tree', he_e1_comp: Optional[FeedForward] = None,
                 e1_ca_comp: Optional[FeedForward] = None,
                 r1r2_comp: Optional[FeedForward] = None,
                 rnn_comp: Optional[Seq2SeqEncoder] = None,
                 ) -> None:
        """
        :param type: tree | tree_shared | rnn | rnn_delim
        :param he_e1_comp:
        :param e1_ca_comp:
        :param r1r2_comp:
        :param rnn_comp:
        """
        super(CtxPathEncoder, self).__init__()
        self._type = type
        if self._type == "tree":
            assert he_e1_comp is not None
            assert e1_ca_comp is not None
            assert r1r2_comp is not None
        elif self._type == "tree_shared":
            assert he_e1_comp is not None
            assert r1r2_comp is not None
        elif self._type in ["rnn", "rnn_delim"]:
            assert rnn_comp is not None
        else:
            raise NotImplementedError
        self._he_e1wh_projector = he_e1_comp
        self._e1_ca_projector = e1_ca_comp
        self._path_projector = r1r2_comp
        self._path_rnn = rnn_comp

    def forward(self, he_rep, e1wh_rep, e1_rep, ca_rep, he_mask_, e1wh_mask_, e1_mask_, ca_mask_):
        """
        compute path representation based on type
        :param he_rep: BCP * 2H
        :param e1wh_rep: BCP * 2H
        :param e1_rep: BCP * 2H
        :param ca_rep: BCP * 2H
        :param he_mask_: BCP * 1
        :param e1wh_mask_: BCP * 1
        :param e1_mask_: BCP * 1
        :param ca_mask_: BCP * 1
        :return: BCP * H
        """
        bcp = he_rep.size(0)
        hdim = he_rep.size(1)
        if torch.cuda.is_available():
            device = he_rep.get_device()
        if self._type in ["tree", "tree_shared"]:
            he_e1wh_rep = self._he_e1wh_projector(torch.cat([he_rep, e1wh_rep], 1))
            he_e1wh_rep = he_e1wh_rep * he_mask_.float()
            if self._type == "tree_shared":
                e1_ca_rep = self._he_e1wh_projector(torch.cat([e1_rep, ca_rep], 1))
            else:
                e1_ca_rep = self._e1_ca_projector(torch.cat([e1_rep, ca_rep], 1))
            e1_ca_rep = e1_ca_rep * ca_mask_.float()
            encoded_paths = self._path_projector(torch.cat([he_e1wh_rep, e1_ca_rep], 1))
            encoded_paths = encoded_paths * ca_mask_.float()  # BCP * H
        elif self._type in ["rnn", "rnn_delim"]:
            if self._type == "rnn_delim":
                if torch.cuda.is_available():
                    ones_float_ = torch.ones([bcp, 1, hdim]).cuda(device=device)
                    ones_long_ = torch.ones([bcp, 1]).long().cuda(device=device)
                else:
                    ones_float_ = torch.ones([bcp, 1, hdim])
                    ones_long_ = torch.ones([bcp, 1]).long()

                path_rnn_in = torch.cat([he_rep.unsqueeze(1),
                                         e1wh_rep.unsqueeze(1),
                                         ones_float_,
                                         e1_rep.unsqueeze(1),
                                         ca_rep.unsqueeze(1),
                                         ones_float_], 1)  # BCP * 6 * 2H
                mask_in = torch.cat([he_mask_, e1wh_mask_, ones_long_,
                                     e1_mask_, ca_mask_, ones_long_], 1)  # BCP * 6
            else:
                path_rnn_in = torch.cat([he_rep.unsqueeze(1),
                                         e1wh_rep.unsqueeze(1),
                                         e1_rep.unsqueeze(1),
                                         ca_rep.unsqueeze(1)], 1)  # BCP * 4 * 2H
                mask_in = torch.cat([he_mask_, e1wh_mask_, e1_mask_, ca_mask_], 1)  # BCP * 4
            encoded_paths = self._path_rnn(path_rnn_in, mask_in)  # BCP * 4/5 * H
            encoded_paths = encoded_paths[:, -1, :]  # BCP * H
            encoded_paths = encoded_paths * ca_mask_.float()  # BCP * H
        else:
            raise NotImplementedError
        return encoded_paths


class JointEncoder(torch.nn.Module, FromParams):
    def __init__(self, seq_encoder: Optional[Seq2SeqEncoder] = None) -> None:
        super(JointEncoder, self).__init__()
        self._seq_encoder = seq_encoder

    def forward(self, doc_encoding, q_encoding, doc_mask, q_mask):
        """

        :param doc_encoding: B * N * T * H
        :param q_encoding: B * U * H
        :param doc_mask: B * N * T
        :param q_mask: B * U
        :return: B * N * T * 2H
        """
        batch_size = doc_encoding.shape[0]
        num_docs = doc_encoding.shape[1]
        num_doc_tokens = doc_encoding.shape[2]
        num_q_tokens = q_encoding.shape[1]
        doc_encoding = doc_encoding.view(batch_size, num_docs * num_doc_tokens, -1)  # B * NT * H
        attn_unnorm = doc_encoding.bmm(q_encoding.transpose(2, 1))  # B * NT * U
        attn = masked_softmax(attn_unnorm, q_mask.unsqueeze(1).expand(attn_unnorm.size()),
                              dim=-1)  # B * NT * U
        aggq = attn.bmm(q_encoding)  # B * NT * H
        attn_t = attn_unnorm.transpose(2, 1).contiguous().view(batch_size,
                                                               -1, num_docs,
                                                               num_doc_tokens)  # B * U * N * T
        attn_t = masked_softmax(attn_t, doc_mask.unsqueeze(1).expand(attn_t.size()), dim=-1)
        attn_t = attn_t.view(batch_size, num_q_tokens, -1)  # B * U * NT
        aggdoc = attn_t.bmm(doc_encoding)  # B * U * H
        aggq2 = attn.bmm(aggdoc)  # B * NT * H
        if self._seq_encoder is not None:
            aggq2 = aggq2.view(batch_size * num_docs, num_doc_tokens, -1)  # BN * T * H
            aggq2 = self._seq_encoder(aggq2, doc_mask.view(batch_size * num_docs, -1))  # BN * T * H

        aggq2 = aggq2.view(doc_encoding.size())  # B * N * T * H
        aggq = aggq.view(doc_encoding.size())  # B * N * T * H
        return torch.cat([aggq, aggq2], -1)


class AttnPooling(torch.nn.Module, FromParams):
    def __init__(self, projector: FeedForward,
                 intermediate_projector: FeedForward = None) -> None:
        super(AttnPooling, self).__init__()
        self._projector = projector
        self._int_proj = intermediate_projector

    def forward(self, xinit: torch.FloatTensor,
                xmask: torch.LongTensor) -> torch.FloatTensor:
        """

        :param xinit: B * T * H
        :param xmask: B * T
        :return: B * H
        """
        if self._int_proj is not None:
            x = self._int_proj(xinit)
            x = x * xmask.unsqueeze(-1)
        else:
            x = xinit
        attn = self._projector(x)  # B * T * 1
        attn = attn.squeeze(-1)  # B * T
        attn = masked_softmax(attn, xmask, dim=-1)
        pooled = attn.unsqueeze(1).bmm(xinit).squeeze(1)  # B * H
        return pooled
