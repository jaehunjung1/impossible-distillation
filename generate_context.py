import json
import os
from argparse import ArgumentParser

import ipdb
import torch
from tqdm import tqdm

from pipeline.orig import OrigGenerator
from pipeline.util import bio_gpt_postprocess


def parse_args_orig():
    args = ArgumentParser(description="GPTWrapper")

    args.add_argument("--device_id", type=int, help="Device to use (0 ~ 7)")
    args.add_argument("--domain", type=str)

    args = args.parse_args()
    args.device = torch.device(f"cuda:{args.device_id}")

    args.total_size = 1000000
    args.save_size = 10000

    return args


if __name__ == "__main__":
    args = parse_args_orig()
    orig_generator = OrigGenerator(args)

    out_filename = f"gen_data/{args.domain}.orig.jsonl"

    if os.path.exists(out_filename):
        raise FileExistsError

    out_list = []
    written_count = 0
    with tqdm(total=args.total_size) as pbar:
        while written_count < args.total_size:
            prefix = orig_generator.generate_prefix()
            batch_pair_list = orig_generator.generate_y_orig(prefix)
            if written_count <= 60000:  # For stage 0, we need to have non-zero length x_l
                batch_pair_list = [pair for pair in batch_pair_list if len(pair[0]) > 0]

            if args.domain == "bio":
                processed_batch_pair_list = [(bio_gpt_postprocess(x_l), bio_gpt_postprocess(y_orig))
                                             for x_l, y_orig in batch_pair_list]
                batch_samples = [json.dumps({
                    "x_l": x_l,
                    "y_orig": y_orig,
                    "x_l_raw": x_l_raw,
                    "y_orig_raw": y_orig_raw,
                }) for (x_l, y_orig), (x_l_raw, y_orig_raw) in zip(processed_batch_pair_list, batch_pair_list)]

            else:
                batch_samples = [json.dumps({
                    "x_l": x_l,
                    "y_orig": y_orig,
                }) for x_l, y_orig in batch_pair_list]

            out_list.extend(batch_samples)
            pbar.update(len(batch_samples))

            if len(out_list) >= args.save_size:
                written_count += len(out_list)

                with open(out_filename, "a") as f:
                    f.write("\n".join(out_list) + "\n")

                out_list = []



