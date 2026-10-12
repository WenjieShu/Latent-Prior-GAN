"""CPU contracts exercised without downloads or large pretrained models."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

import config
from utils.checkpointing import (
    advance_decoder_lr,
    loss_ema_state,
    restore_loss_ema,
    validate_epoch_resume,
)
from utils.decoder_checkpoint import load_decoder_checkpoint, load_tokenizer_state
from utils.pool_resume import validate_shard_resume
from utils.sampling import indexed_inputs, pack_images, to_uint8

torch.set_num_threads(2)


class ConfigContracts(unittest.TestCase):
    def yaml_args(self, text, cli=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text)
            return config.parse_arg(["--config", str(path), *cli], prepare=False)

    def test_yaml_cli_precedence_and_cfg_registry(self):
        args = self.yaml_args(
            "model_type: iMF-M-2\nbatch_size: 4\nuse_cfg: false\n",
            ["--batch_size", "8", "--use_cfg"],
        )
        self.assertEqual((args.batch_size, args.use_cfg, args.cfg_omega), (8, True, 10.5))

    def test_reject_unknown_and_ignored_arguments(self):
        for text in ("stage: stage3", "gan_weight: 0.5", "rec_weight: 1.0", "typo: 3"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.yaml_args(text)

    def test_reject_invalid_yaml_types(self):
        for text in (
            "batch_size: true",
            "batch_size: 1.5",
            "use_cfg: 'false'",
            "epochs: null",
            "dataset_dir: 123",
        ):
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                self.yaml_args(text)

    def test_official_pth_only_and_interval_validation(self):
        for cli in (
            ["--pretrained_imf_pytorch", "converted.pt"],
            ["--cfg_t_min", ".2"],
            ["--cfg_t_min", ".8", "--cfg_t_max", ".2"],
            ["--lr", "nan"],
        ):
            with self.subTest(cli=cli), self.assertRaises(ValueError):
                config.parse_arg(cli, prepare=False)

    def test_missing_prior_fails_before_vae_download_or_allocation(self):
        from models.imf_official import OfficialImfWrapper
        from models.model import LatentPriorGenerator

        args = config.parse_arg(
            ["--pretrained_imf_pytorch", "/nonexistent/official.pth"], prepare=False
        )
        with (
            patch("models.model.build_tokenizer") as tokenizer,
            self.assertRaises(FileNotFoundError),
        ):
            LatentPriorGenerator(args)
        tokenizer.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            OfficialImfWrapper("iMF-B-2", "/nonexistent/official.pth")

    def test_native_formats_and_vae_identity_defaults(self):
        args = config.parse_arg(
            ["--prior_family", "dmf", "--prior_checkpoint", "native.pt"], prepare=False
        )
        self.assertEqual(args.vae_type, "mse")
        args = config.parse_arg(["--prior_family", "meanflow"], prepare=False)
        self.assertEqual(args.vae_type, "ema")
        with self.assertRaises(ValueError):
            config.parse_arg(["--prior_family", "meanflow", "--vae_type", "mse"], prepare=False)

    def test_distinct_training_names_and_examples(self):
        a = config.parse_arg([], prepare=False)
        b = config.parse_arg(["--no-supervision_use_cfg"], prepare=False)
        c = config.parse_arg(["--wo_ps"], prepare=False)
        self.assertEqual(len({a.saver_name_pre, b.saver_name_pre, c.saver_name_pre}), 3)
        for path in Path("configs").glob("*.yaml"):
            with self.subTest(path=path):
                cli = (
                    ["--real_latent_pool", "real.npz", "--generated_latent_pool", "gen.npz"]
                    if path.name.startswith("titok")
                    else []
                )
                config.parse_arg(["--config", str(path), *cli], prepare=False)

    def test_titok_manifest_steps_and_model_identity(self):
        from eval_generation import parse_args

        args = parse_args(["--prior_family", "titok", "--out_dir", "unused"])
        self.assertEqual(
            (args.model_name, args.num_steps, args.cfg_omega, args.seed),
            ("titok-L32", 8, 4.5, 3407),
        )
        args = parse_args(
            ["--config", "configs/dmf_xl.yaml", "--out_dir", "unused", "--batch_size", "3"]
        )
        self.assertEqual(
            (args.prior_family, args.model_name, args.batch_size, args.seed), ("dmf", "dmf", 3, 0)
        )


class DecoderCheckpointContracts(unittest.TestCase):
    def tokenizer(self, family="imf"):
        module = nn.Module()
        if family == "titok":
            module.model = nn.Module()
            for key in ("decoder", "pixel_quantize", "pixel_decoder"):
                setattr(module.model, key, nn.Linear(2, 2))
        else:
            module.vae = nn.Module()
            module.vae.encoder = nn.Linear(2, 2)
            module.vae.decoder = nn.Linear(2, 2)
        return module

    def test_legacy_full_model_keys_restore_both_tokenizer_families(self):
        for family in ("imf", "titok"):
            source, destination = self.tokenizer(family), self.tokenizer(family)
            state = {"module.tokenizer." + key: value for key, value in source.state_dict().items()}
            state["module.diffusion.model.net.unused"] = torch.ones(1)
            info = load_tokenizer_state(destination, state, family)
            self.assertGreater(info["decoder_tensors"], 0)
            for key, value in destination.state_dict().items():
                torch.testing.assert_close(value, source.state_dict()[key])

    def test_legacy_nested_titok_keys_restore_current_tokenizer(self):
        source, destination = self.tokenizer("titok"), self.tokenizer("titok")
        state = {"module.tokenizer.model." + key: value for key, value in source.state_dict().items()}
        info = load_tokenizer_state(destination, state, "titok")
        self.assertEqual(info["decoder_tensors"], 6)
        for key, value in destination.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[key])
        partial = {key: value for key, value in state.items() if ".pixel_quantize." not in key}
        with self.assertRaisesRegex(ValueError, "Incomplete adapted decoder"):
            load_tokenizer_state(destination, partial, "titok")

    def test_reject_empty_partial_and_wrong_shape(self):
        module = self.tokenizer()
        for state in (
            {"diffusion.weight": torch.ones(1)},
            {"vae.decoder.weight": torch.ones(2, 2)},
            {"vae.decoder.weight": torch.ones(3, 2)},
        ):
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                load_tokenizer_state(module, state, "imf")

    def test_reject_family_or_scale_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth.tar"
            for saved in (
                {"prior_family": "dmf"},
                {"prior_family": "imf", "model_type": "iMF-XL-2"},
            ):
                torch.save({"args": saved, "model": self.tokenizer().state_dict()}, path)
                with self.assertRaises(ValueError):
                    load_decoder_checkpoint(self.tokenizer(), path, "imf", "iMF-B-2")

    def test_epoch_lr_and_ema_resume(self):
        parameter = nn.Parameter(torch.ones(1))
        before = torch.optim.AdamW([parameter], lr=8e-6)
        after = torch.optim.AdamW([parameter], lr=1e-5)
        after.load_state_dict(before.state_dict())
        advance_decoder_lr(after, 0.5, 1e-6)
        self.assertEqual(after.param_groups[0]["lr"], 4e-6)
        loss = SimpleNamespace(lecam_ema=SimpleNamespace(logits_real_ema=0.7, logits_fake_ema=-0.3))
        checkpoint = {"loss_ema": loss_ema_state(loss)}
        loss.lecam_ema.logits_real_ema = 0
        restore_loss_ema(loss, checkpoint)
        self.assertEqual(loss.lecam_ema.logits_real_ema, 0.7)
        with self.assertRaises(ValueError):
            validate_epoch_resume({"epoch_complete": False})


class SamplingContracts(unittest.TestCase):
    def test_indexed_seeds_cover_each_sample_once(self):
        shards = [indexed_inputs(23, 42, rank, 4) for rank in range(4)]
        indices = np.concatenate([part[0] for part in shards])
        self.assertEqual(sorted(indices.tolist()), list(range(23)))
        for idx, labels, seeds in shards:
            np.testing.assert_array_equal(labels, idx % 1000)
            np.testing.assert_array_equal(seeds, idx ^ 42)

    def test_official_noise_is_independent_of_batch_shards(self):
        from models.imf_official import BatchGenerator

        full = BatchGenerator("cpu", [42, 43, 44]).randn((3, 4, 8, 8))
        single = torch.cat(
            [BatchGenerator("cpu", [seed]).randn((1, 4, 8, 8)) for seed in [42, 43, 44]]
        )
        self.assertTrue(torch.equal(full, single))

    def test_pack_exact_rgb_count_and_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pixels = to_uint8(torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 3, 1, 1).expand(2, 3, 4, 4))
            for i, sample in enumerate(pixels):
                Image.fromarray(sample).save(path / f"{i:06d}.png")
            pack_images(path, 2, 4, path / "samples.npz")
            with np.load(path / "samples.npz") as data:
                np.testing.assert_array_equal(data["arr_0"], pixels)
            Image.fromarray(pixels[0]).save(path / "stale.png")
            with self.assertRaises(ValueError):
                pack_images(path, 2, 4, path / "samples.npz")

    def test_nonfinite_generation_fails(self):
        with self.assertRaises(ValueError):
            to_uint8(torch.full((1, 3, 2, 2), float("nan")))


class PoolResumeContracts(unittest.TestCase):
    def test_reject_changed_config_and_nondeterministic_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            meta, shard, progress = [
                path / name for name in ("meta.json", "tokens.npy", "progress")
            ]
            signature = {"seed": 42, "labels_hash": "original"}
            self.assertEqual(validate_shard_resume(meta, signature, shard, progress, 3, True), 0)
            np.save(shard, np.ones((3, 2), dtype=np.int16))
            progress.write_text("1")
            self.assertEqual(validate_shard_resume(meta, signature, shard, progress, 3, True), 1)
            with self.assertRaises(ValueError):
                validate_shard_resume(meta, {**signature, "seed": 0}, shard, progress, 3, True)
            with self.assertRaises(ValueError):
                validate_shard_resume(meta, signature, shard, progress, 3, False)
            self.assertEqual(
                validate_shard_resume(meta, {"seed": 0}, shard, progress, 3, True, restart=True), 0
            )

    def test_reject_legacy_shards_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "progress").write_text("1")
            with self.assertRaises(ValueError):
                validate_shard_resume(path / "meta", {}, path / "shard", path / "progress", 3, True)


class PairingContracts(unittest.TestCase):
    def make_pools(self, path, change=None):
        train = path / "imagenet" / "train"
        for i, name in enumerate(("class0", "class1")):
            (train / name).mkdir(parents=True)
            Image.fromarray(np.full((32, 36, 3), 30 + i * 100, dtype=np.uint8)).save(
                train / name / "image.png"
            )
        metadata = dict(
            tokens=np.zeros((2, 32), dtype=np.int16),
            labels=np.array([0, 1], dtype=np.int16),
            image_indices=np.array([0, 1]),
        )
        real = {
            **metadata,
            "sample_paths": np.array(["class0/image.png", "class1/image.png"]),
            "preprocessing": "adm_center_crop_v1",
            "resolution": 256,
            "tokenizer_name": "titok-vql32",
        }
        generated = {**metadata, "generator_name": "titok-L32", "resolution": 256}
        if change:
            change(real, generated)
        np.savez(path / "real.npz", **real)
        np.savez(path / "gen.npz", **generated)
        from data.titok_dataset import PairedARLatentDataset

        return PairedARLatentDataset(path / "real.npz", path / "gen.npz", str(train.parent))

    def test_paired_tokens_use_original_center_cropped_image(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_pools(Path(directory))
            tokens, generated, image = dataset[1]
            self.assertEqual(
                (tokens.shape, generated.shape, image.shape),
                (torch.Size([32]), torch.Size([32]), torch.Size([3, 256, 256])),
            )
            torch.testing.assert_close(image.mean(), torch.tensor(130 / 127.5 - 1))

    def test_reject_missing_indices_wrong_labels_paths_and_tokenizer(self):
        changes = [
            lambda r, g: g.pop("image_indices"),
            lambda r, g: g.update(labels=np.array([1, 0])),
            lambda r, g: r.update(sample_paths=np.array(["wrong/image.png", "class1/image.png"])),
            lambda r, g: g.update(generator_name="titok-B64"),
            lambda r, g: g.update(tokens=np.full((2, 32), 4096)),
            lambda r, g: g.update(tokens=np.zeros((2, 64), dtype=np.int16)),
        ]
        for change in changes:
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                self.make_pools(Path(directory), change)


class TokenizerGradientContracts(unittest.TestCase):
    def test_real_diffusers_vae_freezes_encoder_and_trains_only_decoder(self):
        from diffusers import AutoencoderKL

        from models.tokenizer import Tokenizer

        with tempfile.TemporaryDirectory() as directory:
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
            vae.save_pretrained(directory)
            tokenizer = Tokenizer(
                SimpleNamespace(vae_type="mse", vae_path=directory, decoder_init="pretrained")
            )
            image = torch.randn(2, 3, 8, 8)
            latents = tokenizer.encode(image)
            self.assertFalse(latents.requires_grad)
            expected = (
                tokenizer.vae.encode(image).latent_dist.mean - tokenizer.mean
            ) / tokenizer.std
            torch.testing.assert_close(latents, expected)
            loss = tokenizer.decode(latents).square().mean()
            loss.backward()
            for key, parameter in tokenizer.vae.named_parameters():
                if key.startswith("decoder."):
                    self.assertTrue(parameter.requires_grad)
                    self.assertIsNotNone(parameter.grad)
                else:
                    self.assertFalse(parameter.requires_grad)
                    self.assertIsNone(parameter.grad)


class LossGradientContracts(unittest.TestCase):
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 1)

        def forward(self, images):
            return self.conv(images).mean((2, 3))

    class Perceptual(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, a, b):
            return (a - b).abs().mean()

    def test_generator_then_discriminator_has_no_decoder_gradient_leak(self):
        import models.paired_supervision_losses as w_ps
        import models.titok_losses as titok
        import models.wo_ps_losses as wo_ps

        args = config.parse_arg([], prepare=False)
        for name, module in (("wo_ps", wo_ps), ("w_ps", w_ps), ("titok", titok)):
            with (
                self.subTest(objective=name),
                patch.object(module, "DinoDiscriminator", self.Discriminator),
                patch.object(module, "DiffAugment", lambda x, **kw: x),
            ):
                if name == "w_ps":
                    with patch.object(module, "PerceptualLoss", self.Perceptual):
                        loss_module = module.SupervisedDinoGANLoss(args)
                elif name == "wo_ps":
                    loss_module = module.DinoGANLoss(args)
                else:
                    loss_module = module.ARPairwiseGANLoss()
                decoder = nn.Conv2d(3, 3, 1)
                optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
                disc_optimizer = torch.optim.AdamW(loss_module.discriminator.parameters(), lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    target = torch.randn(2, 3, 8, 8)
                    generated = decoder(torch.randn_like(target))
                    paired = decoder(target * 0.8)
                    if name == "wo_ps":
                        g_loss, _ = loss_module(target, generated, optimizer_idx=0)
                    elif name == "w_ps":
                        g_loss, _ = loss_module(
                            target, generated, paired, optimizer_idx=0, last_layer=decoder.weight
                        )
                    else:
                        g_loss, _ = loss_module(
                            paired, generated, target, optimizer_idx=0, last_layer=decoder.weight
                        )
                    self.assertTrue(torch.isfinite(g_loss))
                    g_loss.backward()
                    self.assertGreater(decoder.weight.grad.abs().sum().item(), 0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    disc_optimizer.zero_grad(set_to_none=True)
                    if name == "wo_ps":
                        d_loss, _ = loss_module(target, generated, optimizer_idx=1)
                    elif name == "w_ps":
                        d_loss, _ = loss_module(target, generated, paired, optimizer_idx=1)
                    else:
                        d_loss, _ = loss_module(
                            paired.detach(), generated.detach(), target, optimizer_idx=1
                        )
                    d_loss.backward()
                    self.assertTrue(all(p.grad is None for p in decoder.parameters()))
                    self.assertGreater(
                        sum(
                            p.grad.abs().sum().item()
                            for p in loss_module.discriminator.parameters()
                        ),
                        0,
                    )
                    disc_optimizer.step()


if __name__ == "__main__":
    unittest.main()
