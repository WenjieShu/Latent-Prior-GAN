"""Public model, evaluation configuration, and process-group contracts."""

import builtins
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import config
from eval_generation import build_models, parse_args
from utils.distributed import init_distributed_mode, is_main_process


class EvaluationConfigContracts(unittest.TestCase):
    def test_training_initialization_and_outputs_do_not_leak_into_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "prior_family: imf\ndecoder_init: random\nbatch_size: 2\nepochs: 9\nresults_dir: training_results\n"
            )
            for extra in ([], ["--decoder_checkpoint", "adapted.pth.tar"]):
                args = parse_args(["--config", str(path), "--out_dir", "unused", *extra])
                self.assertEqual(args.decoder_init, "pretrained")
                self.assertEqual(args.batch_size, 32)
                self.assertFalse(hasattr(args, "epochs"))
                self.assertFalse(hasattr(args, "results_dir"))

    def test_titok_training_yaml_resolves_pair_and_cli_overrides_without_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "prior_family: titok\ntokenizer_name: titok-vqb64\nseed: 123\nbatch_size: 2\n"
            )
            args = parse_args(["--config", str(path), "--out_dir", "unused"])
            self.assertEqual(
                (args.model_name, args.num_steps, args.cfg_omega, args.seed),
                ("titok-B64", 8, 3.0, 123),
            )
            args = parse_args(
                [
                    "--config",
                    str(path),
                    "--out_dir",
                    "unused",
                    "--generator_name",
                    "titok-L32",
                    "--seed",
                    "42",
                    "--batch_size",
                    "7",
                    "--cfg_omega",
                    "2",
                ]
            )
            self.assertEqual(
                (args.model_name, args.cfg_omega, args.seed, args.batch_size),
                ("titok-L32", 2.0, 42, 7),
            )
            with self.assertRaises(ValueError):
                config.parse_arg(["--config", str(path)], prepare=False)

    def test_ignored_backend_options_and_non_mapping_yaml_are_rejected(self):
        cases = [
            (["--prior_checkpoint", "unused.pt"], "prior_checkpoint: unused.pt\n"),
            (["--prior_dtype", "bf16"], "prior_dtype: bf16\n"),
            (
                ["--prior_family", "dmf", "--pretrained_imf_pytorch", "unused.pth"],
                "prior_family: dmf\npretrained_imf_pytorch: unused.pth\n",
            ),
            (
                ["--prior_family", "meanflow", "--prior_dtype", "bf16"],
                "prior_family: meanflow\nprior_dtype: bf16\n",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for cli, yaml_text in cases:
                with (
                    self.subTest(cli=cli),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    config.parse_arg(cli, prepare=False)
                path.write_text(yaml_text)
                with self.subTest(yaml=yaml_text), self.assertRaises(ValueError):
                    config.parse_arg(["--config", str(path)], prepare=False)
            for text in ("false", "[]", "0"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    config.load_yaml_config(path)


class PublicModelContracts(unittest.TestCase):
    class Prior(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(1))
            self.paired_calls = 0

        def generate(self, batch_size, **kwargs):
            return torch.ones(batch_size, 4, 8, 8, device=kwargs["device"]) * 0.1

        def denoise_from_encoder_latent(self, latents, **kwargs):
            self.paired_calls += 1
            return latents * 0.8

    def test_real_vae_pipeline_modes_outputs_and_pretrained_evaluation(self):
        from diffusers import AutoencoderKL

        from models.model import LatentPriorGenerator
        from models.tokenizer import build_tokenizer

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                block_out_channels=(8,),
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=4,
            )
            vae.save_pretrained(path / "vae")
            yaml_path = path / "config.yaml"
            yaml_path.write_text(
                f"prior_family: imf\ndecoder_init: random\nvae_path: {path / 'vae'}\n"
            )
            args = parse_args(["--config", str(yaml_path), "--out_dir", "unused"])
            tokenizers = []

            def load_tokenizer(args):
                tokenizer = build_tokenizer(args)
                tokenizers.append(tokenizer)
                return tokenizer

            with (
                patch("models.model.validate_prior_inputs"),
                patch("models.model.build_prior", side_effect=lambda a: self.Prior()),
                patch("models.model.build_tokenizer", side_effect=load_tokenizer),
            ):
                sample, _ = build_models(args, "cpu")
                self.assertEqual(sample(torch.tensor([0]), torch.tensor([42])).shape, (1, 3, 8, 8))
                for key, value in tokenizers[0].vae.decoder.state_dict().items():
                    torch.testing.assert_close(value, vae.decoder.state_dict()[key], rtol=0, atol=0)
                train_args = config.parse_arg(["--vae_path", str(path / "vae")], prepare=False)
                model = LatentPriorGenerator(train_args)
            model.eval_mode()
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            model.train_mode()
            self.assertFalse(model.diffusion.training)
            self.assertFalse(model.tokenizer.vae.encoder.training)
            self.assertTrue(model.tokenizer.vae.decoder.training)
            self.assertEqual(
                {id(p) for p in model.trainable_parameters()},
                {id(p) for p in model.parameters() if p.requires_grad},
            )
            images, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 1])
            generated, paired = model(images, labels)
            self.assertEqual((generated.shape, paired.shape), (images.shape, images.shape))
            self.assertEqual(model.diffusion.paired_calls, 1)
            generated, paired = model(labels=labels)
            self.assertIsNone(paired)
            self.assertEqual(model.diffusion.paired_calls, 1)
            generated.mean().backward()
            self.assertIsNotNone(model.last_layer.grad)
            self.assertIsNone(model.diffusion.anchor.grad)
            with self.assertRaises(ValueError):
                model(images)

    def test_titok_output_order_preserves_decoder_call_order_and_state_nesting(self):
        from models.titok_model import ARLatentPriorDecoder

        class Tokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.model = nn.Module()
                self.model.model.decoder = nn.Linear(2, 2)
                self.calls = []

            def train_mode(self):
                self.train().requires_grad_(True)

            def eval_mode(self):
                self.eval().requires_grad_(False)

            def latent_to_image(self, tokens):
                self.calls.append(tokens)
                return self.model.model.decoder(tokens)

            def trainable_parameters(self):
                return list(self.parameters())

            @property
            def last_layer(self):
                return self.model.model.decoder.weight

        tokenizer = Tokenizer()
        with patch("models.titok_model.Tokenizer", return_value=tokenizer):
            model = ARLatentPriorDecoder("titok-vql32")
        real_tokens, generated_tokens = torch.ones(2, 2), torch.zeros(2, 2)
        generated, paired = model(real_tokens, generated_tokens)
        self.assertIs(tokenizer.calls[0], real_tokens)
        self.assertIs(tokenizer.calls[1], generated_tokens)
        torch.testing.assert_close(paired, tokenizer.model.model.decoder(real_tokens))
        torch.testing.assert_close(generated, tokenizer.model.model.decoder(generated_tokens))
        self.assertIn("tokenizer.model.model.decoder.weight", model.state_dict())
        model.eval_mode()
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        model.train_mode()
        self.assertTrue(all(p.requires_grad for p in model.trainable_parameters()))


class DistributedAndLoggingContracts(unittest.TestCase):
    def test_torchrun_uses_local_rank_and_retains_family_timeout(self):
        original_print = builtins.print
        for timeout in (None, 7200):
            args = SimpleNamespace()
            with (
                patch.dict(
                    os.environ, {"RANK": "5", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}, clear=True
                ),
                patch("utils.distributed.torch.cuda.set_device") as set_device,
                patch("utils.distributed.dist.init_process_group") as init,
                patch("utils.distributed.dist.barrier"),
            ):
                init_distributed_mode(args, timeout=timeout)
                self.assertEqual((args.rank, args.world_size, args.gpu), (5, 8, 1))
                set_device.assert_called_once_with(1)
                self.assertEqual(
                    init.call_args.kwargs.get("timeout"),
                    timedelta(seconds=7200) if timeout else None,
                )
                self.assertFalse(is_main_process())
        self.assertIs(builtins.print, original_print)

    def test_single_gpu_and_pre_init_slurm_rank(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                init_distributed_mode(SimpleNamespace())
            args = SimpleNamespace()
            init_distributed_mode(args, allow_single=True)
            self.assertEqual((args.distributed, args.world_size, args.gpu), (False, 1, 0))
        with patch.dict(os.environ, {"SLURM_PROCID": "2"}, clear=True):
            self.assertFalse(is_main_process())

    def test_training_error_restores_stdout_and_finishes_process_group(self):
        import train

        with tempfile.TemporaryDirectory() as directory:
            args = config.parse_arg(["--results_dir", directory], prepare=False)
            before = sys.stdout
            with (
                patch.object(train.config, "parse_arg", return_value=args),
                patch.object(
                    train, "train_continuous", side_effect=RuntimeError("training failed")
                ),
                patch.object(train, "finish_distributed") as finish,
                patch.object(train, "is_main_process", return_value=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    train.main([])
                self.assertIs(sys.stdout, before)
                finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
