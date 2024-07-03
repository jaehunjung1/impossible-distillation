import json
import os
import random
import re
from argparse import Namespace
from typing import List, Tuple, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig
import spacy

import ipdb


class OrigGenerator:
    def __init__(self, args: Namespace):
        self.args = args

        self.tokenizer, self.model = self.init_tokenizer_and_model()
        self.generation_config = self.init_generation_config()

        self.spacy_model = spacy.load("en_core_web_sm")

        project_dir = os.path.dirname(os.path.dirname(__file__))

        if self.args.domain != "bio":
            with open(os.path.join(project_dir, f"data/prefix/{self.args.domain}_prefix.json"), "r") as f:
                self.prefix_resource = json.load(f)

    def init_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
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

        return tokenizer, model

    def init_generation_config(self) -> GenerationConfig:
        if self.args.domain == "news":
            bad_words = ["\n\n", "\n"]
            bad_words_ids = self.tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids + \
                self.tokenizer(bad_words, add_prefix_space=False, add_special_tokens=False).input_ids
            generation_config = GenerationConfig(
                max_new_tokens=150, num_return_sequences=100,
                do_sample=True, top_p=0.9, temperature=1.0,
                bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
            )

        elif self.args.domain == "bio":
            bad_words = ["<", ">", "/", "<unk>", "[", "]", "▃"]
            bad_words_ids = self.tokenizer(bad_words, add_special_tokens=False).input_ids
            generation_config = GenerationConfig(
                max_new_tokens=150, num_return_sequences=100,
                do_sample=True, top_p=0.9, temperature=1.0,
                bad_words_ids=bad_words_ids, pad_token_id=self.tokenizer.pad_token_id
            )

        else:
            raise NotImplementedError

        return generation_config

    def generate_prefix(self) -> str:
        if self.args.domain == "news":
            city, country = random.choice(self.prefix_resource["city_country"])
            city = city.upper()
            media = random.choice(self.prefix_resource["media_list"])

            # Select template
            random_seed = random.random()

            if random_seed < 0.5:
                # Only include media name
                prefix = f"({media}) --"
            elif random_seed < 0.75:
                # Include media name and country
                prefix = f"{country} ({media}) --"
            else:
                # Include all
                prefix = f"{city}, {country} ({media}) --"

        elif self.args.domain == "bio":
            topic = random.choice(["Abstract", "Introduction", "Method", "Conclusion"])

            prefix = f"{topic}:"

        else:
            raise NotImplementedError

        return prefix

    def generate_y_orig(self, prefix: Union[str, List[str]]) -> List[Tuple[str, str]]:
        generation_list = self.generate_with_prefix(prefix)

        batch_pair_list = []
        for text_idx, text in enumerate(generation_list):
            if type(prefix) == str:
                sent_list = self.postprocess_generation(prefix, text)
            else:
                sent_list = self.postprocess_generation(prefix[text_idx // self.args.num_return_sequences], text)

            # pair sentences as x_l - y_orig
            pair_list = [(" ".join(sent_list[:i]), sent_list[i]) for i in range(len(sent_list))
                         if self.qualifies_as_y_orig(sent_list[i])]  # leave only the full sentences
            batch_pair_list.extend(pair_list)

        return batch_pair_list

    def generate_with_prefix(self, prefix: Union[str, List[str]]) -> List[str]:
        input_encoding = self.tokenizer(prefix, return_tensors="pt", padding=True).to(self.args.device)

        outputs = self.model.generate(
            **input_encoding,
            generation_config=self.generation_config,
        )

        outputs_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs_str

    def postprocess_generation(self, prefix: str, text: str) -> List[str]:
        if self.args.domain == "news":
            out = text[len(prefix):].strip()
            sent_list = [sent for sent in self.split_sentences(out)]

        elif self.args.domain == "bio":
            out = text[len(prefix):].strip()
            sent_list = [sent for sent in self.split_sentences(out)]

        else:
            raise NotImplementedError

        return sent_list

    def split_sentences(self, text: str) -> List[str]:
        return [str(sent).strip() for sent in self.spacy_model(text).sents]

    def qualifies_as_y_orig(self, text: str) -> bool:
        """Given text, determine whether text qualifies as a legit y_orig"""
        if self.args.domain == "news":
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        elif self.args.domain == "reddit":
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            no_link = "http" not in text
            no_edit = len(re.findall(r'edit([\d\s]+)?:', text.lower())) == 0
            out = default and no_link and no_edit

        elif self.args.domain == "bio":
            default = len(text) >= 3 and "\n" not in text and text[-1] in [".", "?", "!"]
            out = default

        else:
            raise NotImplementedError

        return out


