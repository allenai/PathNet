from typing import List, Dict, Optional
import logging

import torch
from allennlp.common.util import gpu_memory_mb
from allennlp.nn.util import combine_tensors

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def path_encoding(x, y, combine_str, fforward, gate):
    z = combine_tensors(combine_str, [x, y])
    z = fforward(z)
    gatef = gate(z)
    return gatef * z


def get_final_encoder_states(encoder_outputs: torch.Tensor,
                             mask: torch.Tensor,
                             bidirectional: bool = False) -> torch.Tensor:
    """
    Modified over the original Allennlp function

    Given the output from a ``Seq2SeqEncoder``, with shape ``(batch_size, sequence_length,
    encoding_dim)``, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape ``(batch_size, encoding_dim)``.  This is not as simple as
    ``encoder_outputs[:, -1]``, because the sequences could have different lengths.  We use the
    mask (which has shape ``(batch_size, sequence_length)``) to find the final state for each batch
    instance.

    Additionally, if ``bidirectional`` is ``True``, we will split the final dimension of the
    ``encoder_outputs`` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving ``encoder_outputs[:, -1, :encoding_dim/2]`` concated with
    ``encoder_outputs[:, 0, encoding_dim/2:]``.
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1).long() - 1

    # handle -1 cases
    ll_ = (last_word_indices != -1).long()
    last_word_indices = last_word_indices * ll_

    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def seq2vec_seq_aggregate(seq_tensor, mask, aggregate, bidirectional, dim=1):
    """
        Takes the aggregation of sequence tensor
        :param seq_tensor: Batched sequence requires [batch, seq, hs]
        :param mask: binary mask with shape batch, seq_len, 1
        :param aggregate: max, avg, sum
        :param bidirectional: bool(True/False)
        :param dim: The dimension to take the max. for batch, seq, hs it is 1
        :return:
    """
    seq_tensor_masked = seq_tensor * mask.unsqueeze(-1).float()
    aggr_func = None
    if aggregate == "last":
        seq = get_final_encoder_states(seq_tensor, mask, bidirectional)
    elif aggregate == "max":
        aggr_func = torch.max
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "min":
        aggr_func = torch.min
        seq, _ = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "sum":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
    elif aggregate == "avg":
        aggr_func = torch.sum
        seq = aggr_func(seq_tensor_masked, dim=dim)
        seq_lens = torch.sum(mask, dim=dim)  # this returns batch_size, 1
        seq = seq / seq_lens.view([-1, 1])
    else:
        raise NotImplementedError

    return seq


def gather_vectors_using_index(src_tensor, index_tensor) -> torch.FloatTensor:
    """
    Uses the indices in index_tensor to select vectors from src_tensor
    :param src_tensor: batch x N x h
    :param index_tensor:  Indices with dim: batch x C x P x 1
    :return: selected embeddings with dim: batch x C x P x h
    """
    if index_tensor.size()[-1] != 1:
        raise ValueError("Expecting last index to be 1. Found {}".format(index_tensor.size()))
    flat_idx_tensor = index_tensor.view(index_tensor.size(0), -1, 1)  # B * CP * 1

    # B * CP * Th
    expanded_index_size = [x for x in flat_idx_tensor.size()[:-1]] + [src_tensor.size()[-1]]
    expanded_index_tensor = flat_idx_tensor.expand(expanded_index_size).long()  # B * CP * H

    flat_extracted = torch.gather(src_tensor, 1, expanded_index_tensor)  # B * CP * H

    extracted = flat_extracted.view(src_tensor.size(0), index_tensor.size(1),
                                    index_tensor.size(2), -1)  # B * C * P * H
    return extracted


def gather_tensors_using_index(src_tensor, index_tensor) -> torch.FloatTensor:
    """
    Uses the indices in index_tensor to select matrices from src_tensor
    :param src_tensor: batch x N x T x h
    :param index_tensor:  Indices with dim: batch x C x P x 1
    :return: selected embeddings with dim: batch x C x P x T x h
    """
    if index_tensor.size()[-1] != 1:
        raise ValueError("Expecting last index to be 1. Found {}".format(index_tensor.size()))
    flat_idx_tensor = index_tensor.view(index_tensor.size(0), -1, 1, 1)  # B * CP * 1 * 1

    # B * CP * T * h
    expanded_index_tensor = flat_idx_tensor.expand(flat_idx_tensor.shape[:-2]
                                                   + src_tensor.shape[-2:]).long()  # B * CP * T * h

    flat_extracted = torch.gather(src_tensor, 1, expanded_index_tensor)  # B * CP * T * h

    extracted = flat_extracted.view(src_tensor.size(0), index_tensor.size(1),
                                    index_tensor.size(2), src_tensor.size(2), -1)  # B * C * P * T * h
    return extracted


def gather_tensor_masks_using_index(src_tensor_mask, index_tensor) -> torch.FloatTensor:
    """
    Uses the indices in index_tensor to select vectors from src_tensor_mask
    :param src_tensor_mask: batch x N x T
    :param index_tensor:  Indices with dim: batch x C x P x 1
    :return: selected embeddings with dim: batch x C x P x T
    """
    if index_tensor.size()[-1] != 1:
        raise ValueError("Expecting last index to be 1. Found {}".format(index_tensor.size()))

    flat_idx_tensor = index_tensor.view(index_tensor.size(0), -1, 1)  # B * CP * 1

    # B * CP * T
    expanded_index_size = [x for x in flat_idx_tensor.size()[:-1]] + [src_tensor_mask.size()[-1]]
    expanded_index_tensor = flat_idx_tensor.expand(expanded_index_size).long()  # B * CP * T

    flat_extracted = torch.gather(src_tensor_mask, 1, expanded_index_tensor)  # B * CP * T

    extracted_mask = flat_extracted.view(src_tensor_mask.size(0), index_tensor.size(1),
                                         index_tensor.size(2), -1)  # B * C * P * T
    return extracted_mask


def pad_packed_loc_tensors(tensor: torch.FloatTensor,
                           num_cand: int,
                           num_path: int, num_loc: int,
                           track_list: List[List[List[int]]],
                           mask_tensor: torch.LongTensor = None):
    """
    Packing the location-based tensors
    This helps to reduce memory usage
    :param tensor: B * (cpl) * H
    :param num_cand: maximum number of candidates (C)
    :param num_path: maximum number of paths (P)
    :param num_loc: maximum number of locations (L)
    :param track_list: B * (cpl) * 3
    :param mask_tensor: B * (cpl)
    :return: B * C * P * L * H
    """
    batch_size = tensor.size(0)
    cpl = tensor.size(1)
    hdim = tensor.size(-1)
    ind1_tensor = torch.zeros(batch_size, num_cand, num_path, num_loc)  # B * C * P * L
    ind2_tensor = ind1_tensor + cpl  # B * C * P * L
    if torch.cuda.is_available():
        device = tensor.get_device()
        _zeros = torch.zeros([batch_size, 1, hdim]).cuda(device=device)
        _mask_zeros = torch.zeros([batch_size, 1]).long().cuda(device=device)
        ind1_tensor = ind1_tensor.cuda(device=device)
        ind2_tensor = ind2_tensor.cuda(device=device)
    else:
        _zeros = torch.zeros([batch_size, 1, hdim])
        _mask_zeros = torch.zeros([batch_size, 1]).long()
    padded_tensor = torch.cat([tensor, _zeros], dim=1)

    for bidx in range(batch_size):
        tracks = track_list[bidx]  # cpl * 3
        for trackidx, track in enumerate(tracks):
            candidx = track[0]
            pathidx = track[1]
            locidx = track[2]
            ind1_tensor[bidx, candidx, pathidx, locidx] = bidx
            ind2_tensor[bidx, candidx, pathidx, locidx] = trackidx

    output_tensor = padded_tensor[ind1_tensor.long(), ind2_tensor.long()]
    if torch.cuda.is_available():
        output_tensor = output_tensor.cuda(device=device)
    if mask_tensor is not None:
        padded_mask_tensor = torch.cat([mask_tensor, _mask_zeros], dim=1)
        output_mask_tensor = padded_mask_tensor[ind1_tensor.long(), ind2_tensor.long()]
        if torch.cuda.is_available():
            output_mask_tensor = output_mask_tensor.cuda(device=device)
        return output_tensor, output_mask_tensor

    return output_tensor


def pad_packed_loc_tensors_with_docidxs(tensor: torch.FloatTensor,
                                        docidx_tensor: torch.LongTensor,
                                        num_cand: int,
                                        num_path: int, num_loc: int,
                                        track_list: List[List[List[int]]],
                                        mask_tensor: torch.LongTensor = None):
    """
    padding and packing of the location-based tensors with document indices
    :param tensor: B * (cpl) * H
    :param docidx_tensor: B * (cpl) * 1
    :param num_cand: maximum number of candidates (C)
    :param num_path: maximum number of paths (P)
    :param num_loc: maximum number of locations (L)
    :param track_list: B * (cpl) * 3
    :param mask_tensor: B * (cpl)
    :return: B * C * P * L * H
    """
    assert tensor.shape[:2] == docidx_tensor.shape[:2]
    batch_size = tensor.size(0)
    cpl = tensor.size(1)
    hdim = tensor.size(-1)
    ind1_tensor = torch.zeros(batch_size, num_cand, num_path, num_loc)  # B * C * P * L
    ind2_tensor = ind1_tensor + cpl  # B * C * P * L
    if torch.cuda.is_available():
        device = tensor.get_device()
        _zeros = torch.zeros([batch_size, 1, hdim]).cuda(device=device)
        _docidx_zeros = torch.zeros([batch_size, 1, 1]).long().cuda(device=device)
        _mask_zeros = torch.zeros([batch_size, 1]).long().cuda(device=device)
        ind1_tensor = ind1_tensor.cuda(device=device)
        ind2_tensor = ind2_tensor.cuda(device=device)
    else:
        _zeros = torch.zeros([batch_size, 1, hdim])
        _docidx_zeros = torch.zeros([batch_size, 1, 1]).long()
        _mask_zeros = torch.zeros([batch_size, 1]).long()
    padded_tensor = torch.cat([tensor, _zeros], dim=1)
    padded_docidx_tensor = torch.cat([docidx_tensor, _docidx_zeros], dim=1)

    for bidx in range(batch_size):
        tracks = track_list[bidx]  # cpl * 3
        for trackidx, track in enumerate(tracks):
            candidx = track[0]
            pathidx = track[1]
            locidx = track[2]
            ind1_tensor[bidx, candidx, pathidx, locidx] = bidx
            ind2_tensor[bidx, candidx, pathidx, locidx] = trackidx

    output_tensor = padded_tensor[ind1_tensor.long(), ind2_tensor.long()]
    output_docidx_tensor = padded_docidx_tensor[ind1_tensor.long(), ind2_tensor.long()]
    if torch.cuda.is_available():
        output_tensor = output_tensor.cuda(device=device)
        output_docidx_tensor = output_docidx_tensor.cuda(device=device)
    if mask_tensor is not None:
        padded_mask_tensor = torch.cat([mask_tensor, _mask_zeros], dim=1)
        output_mask_tensor = padded_mask_tensor[ind1_tensor.long(), ind2_tensor.long()]
        if torch.cuda.is_available():
            output_mask_tensor = output_mask_tensor.cuda(device=device)
        return output_tensor, output_docidx_tensor, output_mask_tensor

    return output_tensor, output_docidx_tensor
