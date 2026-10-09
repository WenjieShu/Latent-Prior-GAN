"""Evaluate generated samples with an explicitly selected FID/IS protocol."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True)
    p.add_argument("--protocol", required=True, choices=["openai", "torch_fidelity"])
    p.add_argument("--reference", required=True, help="OpenAI VIRTUAL_imagenet256_labeled.npz or torch-fidelity jit_in256_stats.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    for path in (args.samples, args.reference):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.protocol == "openai":
        if args.cpu:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from metric.evaluator import FIDEvaluator
        evaluator = FIDEvaluator()
    else:
        from metric.fidelity import TorchFidelityEvaluator
        evaluator = TorchFidelityEvaluator(cuda=not args.cpu)
    try:
        if args.protocol == "openai":
            fid, inception = evaluator.compute_fid_and_is(args.reference, args.samples)
        else:
            fid, inception = evaluator.compute_fid_and_is(args.samples, args.reference)
    finally:
        evaluator.close()
    result = {"protocol": args.protocol, "reference": str(Path(args.reference).resolve()), "samples": str(Path(args.samples).resolve()), "fid": float(fid), "inception_score": float(inception)}
    manifest = Path(args.samples).parent / "manifest.json"
    if manifest.is_file():
        result["generation"] = json.loads(manifest.read_text())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
