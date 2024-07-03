from argparse import Namespace
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Set
import itertools
import multiprocessing as mp

import ipdb
import networkx as nx
import spacy
import numerizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from pipeline.util import *


class ConFilter:
    def __init__(self, args: Namespace):
        self.args = args

        nli_model_name = "alisawuffles/roberta-large-wanli"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, use_fast=self.args.stage > 0)
        self.nli_tokenizer.model_max_length = 512
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.args.device)
        self.nli_model.eval()

        self.min_length_threshold = 0.8
        self.max_length_threshold = 1.5
        self.nli_threshold = 0.75
        self.reverse_nli_threshold = 0.75
        self.rougeL_threshold = 0.75

        self.nli_max_batch_size = 2048

        self.spacy_model = spacy.load("en_core_web_sm")
        self.rouge = evaluate.load('rouge')

    def filter_all(self, sample: Candidates) -> List[Pair]:
        """
        Choose different filter function depending on the stage
        """
        if self.args.stage == 0:
            return self.filter_bootstrap(sample)
        elif self.args.stage > 0:
            return self.filter_finetuned(sample)
        else:
            raise NotImplementedError

    def filter_finetuned(self, sample: Candidates) -> List[Pair]:
        """
        :param sample: sample with `x_l`, `y_orig` and `y_cons`
        :return: list of pairs
        """
        x_l = sample.x_l
        y_orig = sample.y_orig
        y_cons = sample.y_cons

        # Create pairs of y_orig - y_con
        y_con_pairs = [Pair(x_l=x_l, y_orig=y_orig, y_new=y_con, y_orig_idx=0, y_new_idx=idx)
                       for idx, y_con in enumerate(y_cons)]

        # Filter pairs based on length
        y_con_pairs = [pair for pair in y_con_pairs if self.filter_length(pair)]

        # Filter pairs based on NLI
        y_con_pairs = self.set_and_filter_nli(y_con_pairs)

        # Filter pairs based on reverse-NLI
        y_con_pairs = self.set_and_filter_reverse_nli(y_con_pairs)

        # Filter pairs based on number
        y_con_pairs = self.filter_number(y_con_pairs)

        # Filter pairs based on Rouge-L
        y_con_pairs = self.set_and_filter_overlap(y_con_pairs)

        # Choose best y_con with least reverse-NLI score
        if len(y_con_pairs) > 0:
            y_con_pairs = [min(y_con_pairs, key=lambda x: x.rougeL)]
        else:
            y_con_pairs = []

        return y_con_pairs

    def filter_bootstrap(self, sample: Candidates) -> List[Pair]:
        """
        :param sample: sample with `x_l`, `y_cons`
        :return: list of pairs
        """
        x_l = sample.x_l
        y_cons = sample.y_cons

        # Create pairs of y_cons
        y_con_pair_indices = itertools.combinations(range(len(y_cons)), 2)  # (y_orig, y_summ)
        y_con_pairs = [Pair(x_l=x_l, y_orig=y_cons[idx1], y_new=y_cons[idx2], y_orig_idx=idx1, y_new_idx=idx2)
                       for idx1, idx2 in y_con_pair_indices]

        # Filter pairs based on length
        y_con_pairs = [pair for pair in y_con_pairs if self.filter_length(pair)]

        # Filter pairs based on NLI
        y_con_pairs = self.set_and_filter_nli(y_con_pairs)

        # Filter pairs based on reverse-NLI (filter out those with high y_summ -> y_orig neutral score)
        y_con_pairs = self.set_and_filter_reverse_nli(y_con_pairs)

        # If multiple pairs with same y_orig_idx exist, leave one with lowest reverse-NLI score
        y_con_pairs = self.remove_duplicates(y_con_pairs)

        # Filter pairs based on number
        y_con_pairs = self.filter_number(y_con_pairs)

        # Filter pairs based on Rouge-L
        y_con_pairs = self.set_and_filter_overlap(y_con_pairs)

        return y_con_pairs

    def set_and_filter_nli(self, pairs: List[Pair]) -> List[Pair]:
        """
        Set the NLI scores and filter pairs based on NLI model.
        :param pairs: list of pairs representing (y_orig, y_summ)
        :return: list of boolean representing whether each pair passed NLI filter
        """
        if len(pairs) == 0:
            return []

        y_orig_list = [pair.y_orig for pair in pairs]
        y_summ_list = [pair.y_new for pair in pairs]

        prediction = self.infer_nli(y_orig_list, y_summ_list)
        NLI_scores = prediction[:, 1].tolist()

        out_pairs = []
        for pair, NLI_score in zip(pairs, NLI_scores):
            pair.nli_score = NLI_score

            if pair.nli_score >= self.nli_threshold:
                out_pairs.append(pair)

        return out_pairs

    def set_and_filter_reverse_nli(self, pairs: List[Pair]) -> List[Pair]:
        """
        Set and filter pairs based on reverse NLI score (y_summ => y_orig)
        :param pairs: list of pairs
        """
        if len(pairs) == 0:
            return []

        y_orig_list = [pair.y_orig for pair in pairs]
        y_summ_list = [pair.y_new for pair in pairs]

        prediction = self.infer_nli(y_summ_list, y_orig_list)
        reverse_NLI_scores = prediction[:, 1].tolist()

        out_pairs = []
        for pair, reverse_NLI_score in zip(pairs, reverse_NLI_scores):
            pair.reverse_nli_score = reverse_NLI_score

            if pair.reverse_nli_score >= self.reverse_nli_threshold:
                out_pairs.append(pair)

        return out_pairs

    def remove_duplicates(self, pairs: List[Pair]) -> List[Pair]:
        """
        Remove duplicates in pairs using NLI
        """
        if len(pairs) <= 1:
            return pairs

        ######
        # 1. Remove y_orig duplicates
        ######
        pair_container = PairContainer(pairs)

        # Prepare inputs to run NLI between unique y_origs
        y_orig_indices = pair_container.unique_y_orig_indices()
        y_orig_idx_pairs = list(itertools.permutations(y_orig_indices, 2))
        premise_list = [pair_container.get_y_orig(idx_pair[0]) for idx_pair in y_orig_idx_pairs]
        hypothesis_list = [pair_container.get_y_orig(idx_pair[1]) for idx_pair in y_orig_idx_pairs]

        # Run NLI to find entailments between y_origs
        if len(y_orig_indices) >= 2:
            nli_result = torch.eq(torch.argmax(self.infer_nli(premise_list, hypothesis_list),
                                               dim=-1), 1).nonzero().view(-1).tolist()
            entail_y_orig_idx_pairs = [y_orig_idx_pairs[nli_idx] for nli_idx in nli_result]
        else:
            entail_y_orig_idx_pairs = []

        # Detect connected components of y_origs that entail each other
        connected_components = self.detect_connected_component(nodes=y_orig_indices,
                                                               edges=entail_y_orig_idx_pairs)

        # Retrieve pairs for each connected component and choose one with least reverse NLI neutral score
        new_pairs = []
        for connected_component in connected_components:
            connected_pairs = [pair for idx in connected_component
                               for pair in pair_container.get_pairs_by_y_orig_idx(idx)]

            new_pairs.append(min(connected_pairs, key=lambda x: x.reverse_nli_score))

        if len(new_pairs) <= 1:
            return new_pairs

        ######
        # 2. Remove y_summ duplicates
        ######
        pair_container = PairContainer(new_pairs)

        # Prepare inputs to run NLI between unique y_summs
        y_summ_indices = pair_container.unique_y_summ_indices()
        y_summ_idx_pairs = list(itertools.permutations(y_summ_indices, 2))
        premise_list = [pair_container.get_y_summ(idx_pair[0]) for idx_pair in y_summ_idx_pairs]
        hypothesis_list = [pair_container.get_y_summ(idx_pair[1]) for idx_pair in y_summ_idx_pairs]

        # Run NLI to find entailments between y_summs
        if len(y_summ_indices) >= 2:
            nli_result = torch.eq(torch.argmax(self.infer_nli(premise_list, hypothesis_list),
                                               dim=-1), 1).nonzero().view(-1).tolist()
            entail_y_summ_idx_pairs = [y_summ_idx_pairs[nli_idx] for nli_idx in nli_result]
        else:
            entail_y_summ_idx_pairs = []

        # Detect connected components of y_summs that entail each other
        connected_components = self.detect_connected_component(nodes=y_summ_indices,
                                                               edges=entail_y_summ_idx_pairs)

        # Retrieve pairs for each connected component and choose one with least reverse NLI neutral score
        out_pairs = []
        for connected_component in connected_components:
            connected_pairs = [pair for idx in connected_component
                               for pair in pair_container.get_pairs_by_y_summ_idx(idx)]

            out_pairs.append(min(connected_pairs, key=lambda x: x.reverse_nli_score))

        return out_pairs

    def detect_connected_component(self, nodes: List[int], edges: List[Tuple]) -> List[Set]:
        """
        Given number of nodes and edges, find connected components in the graph
        :param nodes: number of nodes in the graph
        :param edges: list of tuple of edges in the graph
        :return: list of sets of edges representing each connected component
        """
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        return list(nx.connected_components(G))

    def infer_nli(self, premise_list: List[str], hypothesis_list: List[str]) -> torch.LongTensor:
        """
        Infer NLI with given premises and hypotheses. If lists are too long, split and batch-process them.
        :param premise_list: list of premises
        :param hypothesis_list: list of hypotheses
        :return: LongTensor of size (len(premise_list, 3), representing label probabilities
        """

        assert len(premise_list) == len(hypothesis_list), "length of `premise_list` != length of `hypothesis_list`."

        predictions = []
        for start_idx in range(0, len(premise_list), self.nli_max_batch_size):
            batch_premise = premise_list[start_idx:start_idx + self.nli_max_batch_size]
            batch_hypothesis = hypothesis_list[start_idx:start_idx + self.nli_max_batch_size]

            with torch.no_grad():
                input_encoding = self.nli_tokenizer(batch_premise, batch_hypothesis, truncation=True,
                                                    padding=True, return_tensors="pt").to(self.args.device)

                prediction = F.softmax(self.nli_model(**input_encoding).logits, dim=-1)
                predictions.append(prediction)

        if len(predictions) > 0:
            predictions = torch.cat(predictions, dim=0)  # 0: contradiction, 1: entailment, 2: neutral
        else:
            predictions = torch.LongTensor([])
        return predictions

    def filter_length(self, pair: Pair) -> bool:
        min_length = self.min_length_threshold * len(pair.y_orig)
        max_length = self.max_length_threshold * len(pair.y_orig)
        return min_length <= len(pair.y_new) <= max_length

    def set_and_filter_overlap(self, pairs: List[Pair]) -> List[Pair]:
        """
        Compute Rouge-L of each pair and set the Pair attribute
        """
        if len(pairs) > 0:
            if self.args.stage == 0:
                chunk_size = len(pairs) // 4 + 1
                chunk_inputs = [pairs[start_idx: start_idx + chunk_size] for start_idx in range(0, len(pairs), chunk_size)]
                pool = mp.Pool(4)
                results = pool.map(compute_rouge, chunk_inputs)
                pool.close()
                pool.join()

                rougeL_list = [rouge for result in results for rouge in result]
                for pair, rougeL in zip(pairs, rougeL_list):
                    pair.rougeL = rougeL
            else:
                rougeL_list = compute_rouge(pairs)
                for pair, rougeL in zip(pairs, rougeL_list):
                    pair.rougeL = rougeL

        out_pairs = [pair for pair in pairs if pair.rougeL <= self.rougeL_threshold]
        return out_pairs

    def numerize(self, text: str):
        try:
            out = self.spacy_model(text)._.numerize()
        except ValueError:
            out = {}

        return out

    def filter_number(self, pairs: List[Pair]) -> List[Pair]:
        out_pairs = []
        for pair in pairs:
            y_summ_numbers = self.numerize(pair.y_new)
            y_orig_numbers = self.numerize(pair.y_orig)

            if len(y_summ_numbers) == 0 and len(y_orig_numbers) == 0:
                out_pairs.append(pair)
            else:
                y_summ_numbers_set = set(
                    [str(x) for x in y_summ_numbers.keys()] + [str(x) for x in y_summ_numbers.values()]
                )
                y_orig_numbers_set = set(
                    [str(x) for x in y_orig_numbers.keys()] + [str(x) for x in y_orig_numbers.values()]
                )

                y_orig_all_included = all([str(key) in y_summ_numbers_set or str(value) in y_summ_numbers_set
                                           for key, value in y_summ_numbers.items()])
                y_summ_all_included = all([str(key) in y_orig_numbers_set or str(value) in y_orig_numbers_set
                                           for key, value in y_summ_numbers.items()])

                if y_orig_all_included and y_summ_all_included:
                    out_pairs.append(pair)

        return out_pairs





