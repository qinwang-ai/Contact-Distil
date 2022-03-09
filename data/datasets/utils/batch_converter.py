from typing import Sequence, Tuple, List, Union

import torch

RawMSA = Sequence[Tuple[str, str]]

class BatchConverter(object):
    """
    Callable to convert an unprocessed (labels + strings) batch to a processed (labels + tensor) batch.
    """
    def __init__(self, alphabet, data_type="seq",):
        """
        :param alphabet:
        :param data_type: seq, msa
        """
        self.alphabet = alphabet
        self.data_type = data_type

    def __call__(self, raw_data, raw_anns=None):
        if self.data_type == "seq":
            labels, strs, tokens = self.__call_seq__(raw_data)
        else:
            labels, strs, tokens = self.__call_msa__(raw_data)

        # creat a new batch of data tensors
        data = {}
        data["descriptions"] = labels
        data["strings"] = strs
        data["tokens"] = tokens

        # creat a new batch of ann tensors
        if raw_anns is not None:
            num_sample = len(raw_anns)
            anns = {}
            for key in raw_anns[0].keys():
                anns[key + "s"] = []
                for i in range(num_sample):
                    if isinstance(raw_anns[i][key], str):
                        anns[key + "s"].append(raw_anns[i][key])
                    else:
                        anns[key + "s"].append(torch.Tensor(raw_anns[i][key]))
        else:
            anns = None

        return data, anns



    def __call_seq__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for _, seq_str in raw_batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str) in enumerate(raw_batch):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(
                [self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64
            )
            tokens[
            i,
            int(self.alphabet.prepend_bos): len(seq_str)
                                            + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_str) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return labels, strs, tokens

    def __call_msa__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = self.__call_seq__(msa)  #super().__call__(msa)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens
