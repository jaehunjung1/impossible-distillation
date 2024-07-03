import json
import os
import random
from datetime import datetime
from argparse import ArgumentParser, Namespace
from typing import List

import ipdb
import jsonlines
import torch
from tqdm import tqdm

from pipeline.con import ConGenerator
from pipeline.filter import ConFilter


def parse_args():
    parser = ArgumentParser()

    # Arguments that must be specified
    parser.add_argument('--con_stage', type=int)
    parser.add_argument('--con_domain', type=str)
    parser.add_argument('--con_model_name', type=str, help="Model name (e.g. gpt2-xl, wandb run name)")
    parser.add_argument("--device_id", type=int, help="Device to use (0 ~ 7)")

    # Arguments for batch processing
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--shard_start", type=int, default=0)

    # Arguments for saving
    parser.add_argument("--save_size", type=int, default=10)

    args = parser.parse_args()

    # Set device
    args.con_device = torch.device(f"cuda:{args.device_id}")
    args.filter_device = torch.device(f"cuda:{args.device_id}")

    # Set stage
    args.filter_stage = args.con_stage

    if args.con_stage == 0:
        args.batch_size = 1
    else:
        args.batch_size = 20

    # Set input filename
    args.orig_filename = f"gen_data/{args.con_domain}.orig.shuf.jsonl"

    # Check if `orig_filename` is shuffled, and if not, create a shuffled version.
    orig_dir, orig_filename = os.path.split(args.orig_filename)
    if "shuf" not in orig_filename:
        filename_without_ext, ext = os.path.splitext(orig_filename)
        new_orig_filename = os.path.join(orig_dir, f"{filename_without_ext}.shuf{ext}")
        if os.path.exists(new_orig_filename):
            raise FileExistsError(f"{new_orig_filename} already exists.")
        else:
            print(f"Creating {new_orig_filename} from {args.orig_filename}.")
            random.seed(3)

        with jsonlines.open(args.orig_filename, "r") as f:
            data = list(f)
            random.shuffle(data)
            save_to_file(data, new_orig_filename)

        args.orig_filename = new_orig_filename

    # Set save directory
    orig_filename_wo_ext = os.path.splitext(os.path.split(args.orig_filename)[1])[0]
    args.save_dir = f"gen_data/stage{args.con_stage}/{args.con_model_name}/"
    args.save_filename = os.path.join(
        args.save_dir, f"{orig_filename_wo_ext}.{args.shard_start}-{args.shard_start+args.shard_size}.jsonl")

    # Check directory and create one if necessary
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.exists(args.save_filename):
        raise FileExistsError(f"{args.save_filename} already exists.")

    print(f"Saving generations "
          f"from {args.orig_filename}[{args.shard_start}:{args.shard_start+args.shard_size}] "
          f"into {args.save_filename}.")

    return args


def save_to_file(sample_list: List[dict], out_fname: str):
    if len(sample_list) == 0:
        return

    sample_str_list = [json.dumps(sample) for sample in sample_list]
    with open(out_fname, "a") as f:
        f.write("\n".join(sample_str_list) + "\n")


if __name__ == "__main__":
    args = parse_args()
    con_args = Namespace(**{k[4:]: v for k, v in vars(args).items() if k.startswith('con')})
    filter_args = Namespace(**{k[7:]: v for k, v in vars(args).items() if k.startswith('filter')})

    with jsonlines.open(args.orig_filename, "r") as f:
        input_samples = list(f)[args.shard_start:args.shard_start + args.shard_size]

    oom_samples = []

    ########
    # In order to use only 1 GPU per process, for every `save_size` # of input_samples,
    # we unload summ_wrapper then load summ_filter to filter the accumulated `y_summ`s, then save them into file.
    ########
    with tqdm(total=len(input_samples)) as pbar:
        for save_idx in range(0, len(input_samples), args.save_size):
            # 1. Generate with con_generator

            # 1-a) Initialize new con_generator
            con_generator = ConGenerator(con_args)

            # 1-b) Iterate through save_size # of input_samples
            chunked_input_samples = input_samples[save_idx:save_idx + args.save_size]
            unfiltered_candidates_list = []
            for batch_idx in range(0, len(chunked_input_samples), args.batch_size):
                batch_samples = chunked_input_samples[batch_idx:batch_idx + args.batch_size]

                try:
                    candidates = con_generator.generate_y_cons_from_sample(batch_samples)
                except torch.cuda.OutOfMemoryError:
                    # Save samples with OOM error to oom_samples
                    oom_samples.extend(batch_samples)

                    pbar.update(len(batch_samples))
                    continue

                unfiltered_candidates_list.extend(candidates)
                pbar.update(len(candidates))

            # 1-c) End by deleting con_generator
            del con_generator
            torch.cuda.empty_cache()

            # 2. Filter with con_filter

            # 2-a) Initialize new con_filter
            con_filter = ConFilter(filter_args)

            # 2-b) Filter and save them to out_samples
            pairs = []
            for candidates in unfiltered_candidates_list:
                try:
                    pairs.extend(con_filter.filter_all(candidates))
                except torch.cuda.OutOfMemoryError:
                    continue

            # 2-c) End by deleting con_filter
            del con_filter
            torch.cuda.empty_cache()

            # 3. Write the out_samples on a file
            save_to_file([pair.to_dict(with_score=True) for pair in pairs], args.save_filename)
