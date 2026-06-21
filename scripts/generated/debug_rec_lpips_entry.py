"""Run a short rec+LPIPS training debug without changing the train loop."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import post_train_rec_lpips

ORIGINAL_BUILD_DATALOADER = post_train_rec_lpips.build_dataloader


class LimitedLoader:
    def __init__(self, loader, max_steps):
        self.loader = loader
        self.max_steps = max_steps

    def __iter__(self):
        for step, batch in enumerate(self.loader):
            if step >= self.max_steps:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_steps)


def limited_build_dataloader(args, split="train"):
    loader, sampler, size = ORIGINAL_BUILD_DATALOADER(args, split=split)
    if split == "train":
        loader = LimitedLoader(loader, args.debug_steps)
    return loader, sampler, size


if __name__ == "__main__":
    args = config.parse_arg()
    post_train_rec_lpips.build_dataloader = limited_build_dataloader
    post_train_rec_lpips.main_worker(args)
