import os
import re
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Iterable, Union

import ipdb
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig, \
    AutoModelForSeq2SeqLM

from pipeline.util import Candidates, bio_gpt_postprocess


class ConGenerator:
    def __init__(self, args: Namespace):
        self.args = args

        self.tokenizer, self.model = self.init_tokenizer_and_model()
        self.generation_config = self.init_generation_config()

        self.spacy_model = spacy.load("en_core_web_sm")

    def init_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        if self.args.stage == 0:
            if self.args.domain == "news":
                tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", use_fast=False)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"

                model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(self.args.device)

            elif self.args.domain == "bio":
                tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-large", use_fast=False)
                tokenizer.padding_side = "left"

                model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-large").to(self.args.device)

            else:
                raise NotImplementedError

        elif self.args.stage > 0:  # T5
            tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-large")

            model_name = f"{self.args.model_name}"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.args.device)

        else:
            raise NotImplementedError

        return tokenizer, model

    def init_generation_config(self) -> GenerationConfig:
        if self.args.stage == 0:
            if self.args.domain == "news":
                bad_words = ["\n\n", "\n"]
                bad_word_ids = self.tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids + \
                    self.tokenizer(bad_words, add_prefix_space=False, add_special_tokens=False).input_ids
                generation_config = GenerationConfig(
                    max_new_tokens=100, num_return_sequences=100,
                    do_sample=True, top_p=0.9, temperature=0.5, no_repeat_ngram_size=3,
                    bad_words_ids=bad_word_ids, pad_token_id=self.tokenizer.pad_token_id
                )

            elif self.args.domain == "bio":
                bad_words = ["<", ">", "/", "<unk>", "[", "]", "▃"]
                bad_words_ids = self.tokenizer(bad_words, add_special_tokens=False).input_ids
                generation_config = GenerationConfig(
                    max_new_tokens=100, num_return_sequences=100,
                    do_sample=True, top_p=0.9, temperature=0.5, no_repeat_ngram_size=3,
                    bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
                )

            else:
                raise NotImplementedError

        elif self.args.stage > 0:
            generation_config = GenerationConfig(
                max_new_tokens=100, num_return_sequences=10,
                do_sample=True, top_p=0.9, temperature=0.7, no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id
            )

        else:
            raise NotImplementedError

        return generation_config

    def generate_y_cons_from_sample(self, samples: List[dict]) -> List[Candidates]:
        """
        Batched version of the full pipeline generating `y_con`s
        :param samples: list of samples with "x_l", "y_orig"
        :return: out_sample with "x_l", "y_orig" and "y_cons"
        """
        if self.args.stage == 0:
            assert len(samples) == 1, "size of `samples` should be 1 for stage 0."

            if self.args.domain == "bio":
                prefixes = [sample["x_l_raw"] for sample in samples]

            else:
                prefixes = [sample["x_l"] for sample in samples]

        else:
            prefixes = [f"{sample['y_orig']}" for sample in samples]

        generations = self.generate_with_prefix(prefixes)
        num_gen_per_sample = len(generations) // len(samples)

        candidates = []
        for idx, (prefix, sample) in enumerate(zip(prefixes, samples)):
            generations_for_this_prefix = generations[num_gen_per_sample * idx:num_gen_per_sample * (idx + 1)]
            y_cons_for_this_prefix = [self.postprocess_generation(prefix, text) for text in generations_for_this_prefix]
            y_cons_for_this_prefix = [y_con for y_con in y_cons_for_this_prefix if self.qualifies_as_y_con(y_con)]

            candidates.append(
                Candidates(x_l=sample["x_l"], y_orig=sample["y_orig"], y_cons=y_cons_for_this_prefix)
            )

        return candidates

    def generate_with_prefix(self, prefix: Union[str, List[str]]) -> List[str]:
        input_encoding = self.tokenizer(prefix, return_tensors="pt", truncation=True, padding=True).to(self.args.device)

        outputs = self.model.generate(
            **input_encoding,
            generation_config=self.generation_config,
        )

        outputs_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs_str

    def postprocess_generation(self, prefix: str, text: str) -> str:
        if self.args.stage == 0:
            out = text[len(prefix):].strip()
            if self.args.domain == "bio":
                out = bio_gpt_postprocess(out)

            sent_list = self.split_sentences(out)

            if len(sent_list) > 0:
                postprocessed = sent_list[0]
            else:
                postprocessed = ''

        elif self.args.stage > 0:
            postprocessed = text.strip()

        else:
            raise NotImplementedError

        return postprocessed

    def split_sentences(self, text: str) -> List[str]:
        return [str(sent).strip() for sent in self.spacy_model(text).sents]

    def qualifies_as_y_con(self, text: str) -> bool:
        """Given text, determine whether text qualifies as a legit y_con"""
        if self.args.domain in ["news"]:
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        elif self.args.domain in ["bio"]:
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        else:
            raise NotImplementedError

        return out

