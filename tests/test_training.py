"""Unified-entrypoint regressions using real losses, optimizers, and checkpoints.

Only pretrained feature/model networks and CUDA/DDP transport are substituted.
These checks do not claim to validate multi-process DDP or pretrained numerics.
"""

import io
import tempfile
import unittest
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

import config
import train
from utils.checkpointing import validate_training_objective


def tiny_discriminator():
    return nn.Sequential(nn.Conv2d(3, 1, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(1))


class TinyPerceptual(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, a, b):
        return (a - b).abs().mean()


class TinyTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv2d(3, 3, 1)

    def trainable_parameters(self):
        return list(self.decoder.parameters())

    @property
    def last_layer(self):
        return self.decoder.weight

    def train_mode(self):
        pass


class TinyGenerator(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.tokenizer = TinyTokenizer()
        self.diffusion = nn.Identity()
        self.paired_calls = 0

    def train_mode(self):
        self.train()

    def trainable_parameters(self):
        return self.tokenizer.trainable_parameters()

    @property
    def last_layer(self):
        return self.tokenizer.last_layer

    def forward(self, images=None, labels=None):
        batch = len(labels)
        generated = self.tokenizer.decoder(torch.full((batch, 3, 4, 4), 0.3))
        if images is None:
            return generated, None
        self.paired_calls += 1
        return generated, self.tokenizer.decoder(images * 0.8)


class TinyTiTok(TinyGenerator):
    def forward(self, real, generated):
        paired = self.tokenizer.decoder(real)
        return self.tokenizer.decoder(generated), paired


class CPUTransport:
    """Use real torch operations while substituting CUDA device transport."""

    cuda = SimpleNamespace(
        is_available=lambda: True, set_device=lambda _: None, synchronize=lambda: None
    )

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, *args):
        return torch.device("cpu")

    def autocast(self, **kwargs):
        return nullcontext()


class LocalDDP(nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class UnifiedTrainingContracts(unittest.TestCase):
    def test_yaml_objective_cli_override_and_family_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "args.yaml"
            path.write_text("wo_ps: true\nprior_family: imf\n")
            args = config.parse_arg(["--config", str(path)], prepare=False)
            self.assertEqual(args.objective, "wo_ps")
            args = config.parse_arg(["--config", str(path), "--no-wo_ps"], prepare=False)
            self.assertEqual(args.objective, "w_ps")
            path.write_text(
                "prior_family: titok\nreal_latent_pool: real\ngenerated_latent_pool: gen\n"
            )
            args = config.parse_arg(["--config", str(path), "--batch_size", "3"], prepare=False)
            self.assertEqual(
                (args.prior_family, args.batch_size, args.seed, args.weight_decay),
                ("titok", 3, 3407, 0),
            )
            self.assertFalse(hasattr(args, "vae_type"))
            path.write_text("prior_family: titok\n")
            args = config.parse_arg(["--config", str(path), "--prior_family", "imf"], prepare=False)
            self.assertEqual(args.prior_family, "imf")

    def test_family_options_and_unverified_ablation_are_rejected(self):
        pool_args = [
            "--prior_family",
            "titok",
            "--real_latent_pool",
            "real",
            "--generated_latent_pool",
            "gen",
        ]
        for extra in (
            ["--wo_ps"],
            ["--lr_decay", "nan"],
            ["--pairwise_weight", "-1"],
            ["--save_interval", "2"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                config.parse_arg(pool_args + extra, prepare=False)
        with self.assertRaises(ValueError):
            config.parse_arg(["--prior_family", "titok"], prepare=False)
        for cli in (pool_args + ["--num_steps", "1"], ["--real_latent_pool", "real"]):
            with (
                self.subTest(cli=cli),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                config.parse_arg(cli, prepare=False)

    def test_legacy_objective_resume_is_compatible_but_cannot_switch_ps(self):
        for saved, wo_ps in (
            ({"objective": "pure"}, True),
            ({"objective": "supervised"}, False),
            ({"objective": "wo_ps"}, True),
            ({"wo_ps": False}, False),
        ):
            for metadata in (saved, SimpleNamespace(**saved)):
                validate_training_objective({"args": metadata}, SimpleNamespace(wo_ps=wo_ps))
                with self.assertRaises(ValueError):
                    validate_training_objective(
                        {"args": metadata}, SimpleNamespace(wo_ps=not wo_ps)
                    )
        validate_training_objective({}, SimpleNamespace(wo_ps=False))

    def run_loop(self, directory, wo_ps=False, titok=False, epochs=1, resume="", debug_steps=0):
        args_cli = [
            "--checkpoint_dir",
            str(directory),
            "--epochs",
            str(epochs),
            "--resume",
            str(resume),
            "--workers",
            "0",
            "--batch_size",
            "2",
            "--debug_steps",
            str(debug_steps),
            "--saver_name_pre",
            "test",
        ]
        if wo_ps:
            args_cli += ["--wo_ps"]
        if titok:
            args_cli += [
                "--prior_family",
                "titok",
                "--real_latent_pool",
                "real",
                "--generated_latent_pool",
                "gen",
            ]
        args = config.parse_arg(args_cli, prepare=False)
        args.gpu = "cpu"
        args.distributed = False
        images = torch.linspace(-0.8, 0.8, 96).reshape(2, 3, 4, 4)
        model = TinyTiTok() if titok else TinyGenerator()
        events = []
        original_loss = train.continuous_loss

        def recorded_loss(*a, **kw):
            events.append(a[4])
            return original_loss(*a, **kw)

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(train, "torch", CPUTransport()))
            stack.enter_context(patch.object(train, "DDP", LocalDDP))
            stack.enter_context(
                patch.object(train.nn.SyncBatchNorm, "convert_sync_batchnorm", lambda x: x)
            )
            if titok:
                import models.titok_losses as losses

                stack.enter_context(
                    patch.object(train, "init_distributed_mode", lambda a, **kw: None)
                )
                stack.enter_context(
                    patch(
                        "data.titok_dataset.PairedARLatentDataset",
                        return_value=TensorDataset(images * 0.8, images * 0.3, images),
                    )
                )
                stack.enter_context(
                    patch("models.titok_model.ARLatentPriorDecoder", return_value=model)
                )
            else:
                import models.paired_supervision_losses as paired
                import models.wo_ps_losses as without

                losses = without if wo_ps else paired
                stack.enter_context(patch.object(paired, "PerceptualLoss", TinyPerceptual))
                stack.enter_context(patch.object(train, "init_distributed_mode", lambda a: None))
                stack.enter_context(patch.object(train, "LatentPriorGenerator", return_value=model))
                stack.enter_context(
                    patch.object(
                        train,
                        "build_dataloader",
                        return_value=([(images, torch.tensor([0, 1]))] * 2, None, 4),
                    )
                )
                stack.enter_context(patch.object(train, "continuous_loss", recorded_loss))
            stack.enter_context(patch.object(losses, "DinoDiscriminator", tiny_discriminator))
            stack.enter_context(patch.object(losses, "DiffAugment", lambda x, **kw: x))
            (train.train_titok if titok else train.train_continuous)(args)
        return model, events

    def assert_checkpoints_equal(self, a, b):
        for key in ("model", "discriminator"):
            self.assertEqual(set(a[key]), set(b[key]))
            for name in a[key]:
                torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
        self.assertEqual(a["loss_ema"], b["loss_ema"])
        self.assertEqual(a["global_step"], b["global_step"])
        for key in ("optimizer", "optimizer_disc"):
            self.assertEqual(a[key]["param_groups"], b[key]["param_groups"])
            for idx, state in a[key]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b[key]["state"][idx][name], rtol=0, atol=0)

    def test_continuous_ps_routing_update_order_and_exact_epoch_resume(self):
        for wo_ps in (False, True):
            with self.subTest(wo_ps=wo_ps), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                torch.manual_seed(123)
                model, events = self.run_loop(path / "full", wo_ps=wo_ps, epochs=2)
                self.assertEqual(events, [0, 1] * 4)
                self.assertEqual(model.paired_calls, 0 if wo_ps else 4)
                torch.manual_seed(123)
                self.run_loop(path / "split", wo_ps=wo_ps)
                checkpoint = path / "split/checkpoint-test-epoch001.pth.tar"
                self.run_loop(path / "split", wo_ps=wo_ps, epochs=2, resume=checkpoint)
                a = torch.load(path / "full/checkpoint-test-epoch002.pth.tar", weights_only=False)
                b = torch.load(path / "split/checkpoint-test-epoch002.pth.tar", weights_only=False)
                self.assertTrue(b["epoch_complete"])
                self.assertEqual(b["args"]["wo_ps"], wo_ps)
                self.assert_checkpoints_equal(a, b)
                with self.assertRaises(ValueError):
                    self.run_loop(path / "wrong", wo_ps=not wo_ps, epochs=2, resume=checkpoint)

    def test_titok_unified_loop_lr_decay_and_exact_epoch_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.manual_seed(123)
            self.run_loop(path / "full", titok=True, epochs=2)
            torch.manual_seed(123)
            self.run_loop(path / "split", titok=True)
            self.run_loop(
                path / "split",
                titok=True,
                epochs=2,
                resume=path / "split/checkpoint-test-epoch001.pth.tar",
            )
            a = torch.load(path / "full/checkpoint-test-epoch002.pth.tar", weights_only=False)
            b = torch.load(path / "split/checkpoint-test-epoch002.pth.tar", weights_only=False)
            self.assertEqual(b["optimizer"]["param_groups"][0]["lr"], 5e-6)
            self.assertEqual(b["optimizer_disc"]["param_groups"][0]["lr"], 1e-5)
            self.assert_checkpoints_equal(a, b)

    def test_partial_epoch_checkpoint_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.run_loop(path, debug_steps=1)
            checkpoint = path / "checkpoint-test-epoch001.pth.tar"
            saved = torch.load(checkpoint, weights_only=False)
            self.assertFalse(saved["epoch_complete"])
            with self.assertRaises(ValueError):
                self.run_loop(path, epochs=2, resume=checkpoint)


if __name__ == "__main__":
    unittest.main()
