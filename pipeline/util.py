import re
from dataclasses import dataclass
from typing import List

import evaluate
import ipdb

rouge = evaluate.load('rouge')


@dataclass
class Candidates:
    x_l: str
    y_orig: None
    y_cons: List[str]


@dataclass
class Pair:
    x_l: str
    y_orig: str
    y_new: str
    y_orig_idx: int
    y_new_idx: int
    rougeL: float = 100.
    nli_score: float = 0.  # y_orig => y_summ NLI entailment
    reverse_nli_score: float = 0.  # y_summ => y_orig NLI entailment

    def __eq__(self, other):
        return self.y_orig_idx == other.y_orig_idx and self.y_new_idx == other.y_new_idx

    def hash(self, other):
        return hash((self.y_orig_idx, self.y_new_idx))

    def to_dict(self, with_score: bool = True) -> dict:
        if with_score:
            out_dict = {
                "x_l": self.x_l,
                "y_orig": self.y_orig,
                "y_summ": self.y_new,
                "rougeL": self.rougeL,
                "nli": self.nli_score,
                "reverse_nli": self.reverse_nli_score
            }
        else:
            out_dict = {
                "x_l": self.x_l,
                "y_orig": self.y_orig,
                "y_summ": self.y_new,
            }

        return out_dict


class PairContainer:
    def __init__(self, pair_list: List[Pair]):
        self.pair_list = pair_list

        self.y_orig_idx_to_pair = {}
        self.y_orig_idx_to_y_orig = {}
        self.y_summ_idx_to_pair = {}
        self.y_summ_idx_to_y_summ = {}
        for pair in self.pair_list:
            if pair.y_orig_idx not in self.y_orig_idx_to_pair.keys():
                self.y_orig_idx_to_pair[pair.y_orig_idx] = []
                self.y_orig_idx_to_y_orig[pair.y_orig_idx] = pair.y_orig
            self.y_orig_idx_to_pair[pair.y_orig_idx].append(pair)

            if pair.y_new_idx not in self.y_summ_idx_to_pair.keys():
                self.y_summ_idx_to_pair[pair.y_new_idx] = []
                self.y_summ_idx_to_y_summ[pair.y_new_idx] = pair.y_new
            self.y_summ_idx_to_pair[pair.y_new_idx].append(pair)

    def get_pairs_by_y_orig_idx(self, y_orig_idx: int) -> List[Pair]:
        return self.y_orig_idx_to_pair[y_orig_idx]

    def get_pairs_by_y_summ_idx(self, y_summ_idx: int) -> List[Pair]:
        return self.y_summ_idx_to_pair[y_summ_idx]

    def get_y_orig(self, y_orig_idx: int) -> str:
        return self.y_orig_idx_to_y_orig[y_orig_idx]

    def get_y_summ(self, y_summ_idx: int) -> str:
        return self.y_summ_idx_to_y_summ[y_summ_idx]

    def unique_y_orig_indices(self) -> List[int]:
        return list(self.y_orig_idx_to_pair.keys())

    def unique_y_summ_indices(self) -> List[int]:
        return list(self.y_summ_idx_to_pair.keys())


def compute_rouge(pairs: List[Pair]):
    predictions = [pair.y_new for pair in pairs]
    references = [pair.y_orig for pair in pairs]
    rougeL_list = rouge.compute(predictions=predictions, references=references,
                                use_aggregator=False)['rougeL']

    return rougeL_list


def bio_gpt_postprocess(text: str):
    """
    Post-process digit tokens in BioGPT generations
    """
    digit_matches = list(re.finditer(r"(\d\.|\d,|\d)", text))
    space_digit_indices = [match.start() for match in list(re.finditer(r" \d", text))]

    space_indices = [-1, len(text)]
    for match in digit_matches:
        if match.end() in space_digit_indices:
            space_indices.append(match.end())
    space_indices.sort()

    # Remove all index in space_indices
    sub_str_indices = [(space_indices[i] + 1, space_indices[i + 1]) for i in range(len(space_indices) - 1)]

    out_text = "".join([text[start_idx:end_idx] for start_idx, end_idx in sub_str_indices])

    return out_text




