"""Verify cached image/token pairs and inspect fixed-token decoder diversity."""

import argparse
import json
import os

import numpy as np
import torch
from torchvision.utils import save_image

from data.titok_dataset import PairedARLatentDataset
from models.discrete_tokenizer import Tokenizer
from utils.decoder_checkpoint import load_decoder_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_pool", required=True)
    parser.add_argument("--generated_pool", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--tokenizer_name", choices=["titok-vql32", "titok-vqb64"], default="titok-vql32")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--strict_pairing", action="store_true",
                        help="fail if diagnostic re-encoding differs from cached tokens")
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = PairedARLatentDataset(args.real_pool, args.generated_pool, args.dataset_dir)
    model = Tokenizer(args.tokenizer_name, args.tokenizer_ckpt).cuda()
    model.eval_mode()
    indices = np.linspace(0, len(dataset) - 1, min(64, len(dataset)), dtype=int)
    generated = torch.from_numpy(dataset.generated_tokens[indices].astype(np.int64)).cuda()
    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        mismatch_rows = []
        for index in indices[:8]:
            real_tokens, _, image = dataset[int(index)]
            encoded = model.extract_latents(image[None].cuda()).cpu().reshape(-1)
            if not torch.equal(encoded, real_tokens):
                mismatch_rows.append(int(index))
        if mismatch_rows:
            message = (
                "Diagnostic re-encoding differs from cached tokens at rows "
                f"{mismatch_rows}; cached tokens remain the training target."
            )
            if args.strict_pairing:
                raise ValueError(message)
            print(f"[PAIRING_WARNING] {message}", flush=True)
        baseline = torch.cat([model.latent_to_image(batch).cpu() for batch in generated.split(4)])
        result = {
            "diagnostic_pairs": min(8, len(indices)),
            "fixed_samples": len(indices),
            "diagnostic_reencode_mismatch_rows": mismatch_rows,
            "diagnostic_reencode_mismatch_count": len(mismatch_rows),
        }
        save_image((baseline + 1) / 2, os.path.join(args.out_dir, "baseline.png"), nrow=8)
        result["baseline_between_pixel_std"] = baseline.std(dim=0).mean().item()
        if args.checkpoint:
            load_decoder_checkpoint(model, args.checkpoint, "titok", args.tokenizer_name)
            model.eval_mode()
            adapted = torch.cat([model.latent_to_image(batch).cpu() for batch in generated.split(4)])
            if not torch.isfinite(adapted).all():
                raise ValueError("Non-finite adapted output")
            result["checkpoint"] = args.checkpoint
            result["adapted_between_pixel_std"] = adapted.std(dim=0).mean().item()
            result["diversity_ratio"] = (result["adapted_between_pixel_std"] /
                                         max(result["baseline_between_pixel_std"], 1e-8))
            save_image((adapted + 1) / 2, os.path.join(args.out_dir, "adapted.png"), nrow=8)
        with open(os.path.join(args.out_dir, "check.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result), flush=True)
        if result.get("diversity_ratio", 1.0) < 0.1:
            raise RuntimeError("Severe diversity collapse: retain checkpoint and skip 50K evaluation")


if __name__ == "__main__":
    main()
