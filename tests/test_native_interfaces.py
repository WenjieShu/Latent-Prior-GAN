"""Small native architectures exercise sampling/CFG without pretrained downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import config


class NativeInterfaceContracts(unittest.TestCase):
    def tiny_imf(self, **kwargs):
        from models.imf_torch.imfDiT import imfDiT
        return imfDiT(hidden_size=32, depth=2, num_heads=4, aux_head_depth=1, **kwargs)

    def test_official_wrapper_matches_native_guided_and_unguided_sampling(self):
        from models.imf_torch import imfDiT
        from models.imf_torch.imf import iMeanFlow
        from models.imf_official import OfficialImfWrapper, BatchGenerator
        with patch.object(imfDiT, "imfDiT_B_2", self.tiny_imf), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official.pth"
            native = iMeanFlow("imfDiT_B_2").eval()
            torch.save(native.state_dict(), path)
            wrapper = OfficialImfWrapper("iMF-B-2", str(path)).eval()
            labels, seeds = torch.tensor([0, 1]), torch.tensor([42, 43])
            for cfg, omega, lo, hi in ((True, 8., .4, .65), (False, 1., 0., 1.)):
                actual = wrapper.generate(2, labels, num_steps=1, use_cfg=cfg, device="cpu", seeds=seeds)
                expected = native.generate(2, BatchGenerator("cpu", seeds), 1, omega, lo, hi, labels)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            wrapper.train()
            self.assertFalse(wrapper.training)
            self.assertFalse(wrapper.model.training)

    def test_cfg_override_reaches_generation_and_paired_denoising(self):
        from models.imf_torch import imfDiT
        from models.imf_torch.imf import iMeanFlow
        from models.priors import build_prior
        with patch.object(imfDiT, "imfDiT_B_2", self.tiny_imf), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official.pth"
            native = iMeanFlow("imfDiT_B_2").eval()
            torch.save(native.state_dict(), path)
            args = config.parse_arg(["--pretrained_imf_pytorch", str(path), "--cfg_omega", "3", "--cfg_t_min", ".1", "--cfg_t_max", ".9"], prepare=False)
            prior = build_prior(args)
            latent = torch.randn(2, 4, 32, 32)
            labels = torch.tensor([0, 1])
            guided = prior.denoise(latent, .2, labels, use_cfg=True)
            expected = native.sample_one_step(latent, labels, 0, torch.tensor([.2, 0.]), torch.tensor(3.), torch.tensor(.1), torch.tensor(.9))
            torch.testing.assert_close(guided, expected)
            # Paired fixed-noise mode applies the same native equation after interpolation.
            torch.manual_seed(123)
            actual = prior.denoise_from_encoder_latent(latent, .2, labels=labels, use_cfg=True, fixed_noise=True, minimum_noise_level=.2)
            torch.manual_seed(123)
            noisy = .8 * latent + .2 * torch.randn_like(latent)
            expected = native.sample_one_step(noisy, labels, 0, torch.tensor([.2, 0.]), torch.tensor(3.), torch.tensor(.1), torch.tensor(.9))
            torch.testing.assert_close(actual, expected)

    def test_dmf_seed_tensor_and_latent_bridge(self):
        from models.priors.dmf import DmfPrior
        from models.priors.registry import prior_defaults
        from models.priors.latent_bridge import lpg_to_dmf, dmf_to_lpg
        prior = DmfPrior.__new__(DmfPrior)
        nn.Module.__init__(prior)
        prior.defaults = prior_defaults("dmf")
        prior.prior_dtype = torch.float32
        prior.model = nn.Linear(1, 1)
        prior._euler_sampler = lambda **kw: kw["latents"]
        prior.num_classes = 1000
        seeds, labels = torch.tensor([42, 43]), torch.tensor([0, 1])
        full = prior.generate(2, labels, device="cpu", seeds=seeds)
        single = torch.cat([prior.generate(1, labels[i:i+1], device="cpu", seeds=seeds[i:i+1]) for i in range(2)])
        torch.testing.assert_close(full, single, rtol=0, atol=0)
        native = torch.randn(2, 4, 32, 32)
        torch.testing.assert_close(lpg_to_dmf(dmf_to_lpg(native)), native)

    def test_meanflow_native_shape_cfg_and_seed_contract(self):
        from models.priors.alphaflow import AlphaFlowPrior
        from models.priors.latent_bridge import raw_to_lpg
        from omegaconf import OmegaConf
        class NativeNet(nn.Module):
            input_shape = (1, 4, 8, 8)
            def construct_cond_from_labels(self, labels):
                return labels
            def normalize_latents(self, latents):
                return latents
            def denormalize_latents(self, latents):
                return latents
            def forward(self, z, **kw):
                return torch.full_like(z, .1 if kw.get("force_label_dropout") else .2)
        prior = AlphaFlowPrior.__new__(AlphaFlowPrior)
        nn.Module.__init__(prior)
        prior.net = NativeNet()
        prior.num_classes = 1000
        prior.sampling = OmegaConf.create({"cfg": {"scale": 2., "t_min": 0., "t_max": 1.}, "sigma_noise": 1.})
        seeds, labels = torch.tensor([42, 43]), torch.tensor([0, 1])
        guided = prior.generate(2, labels, device="cpu", seeds=seeds)
        noise = torch.stack([torch.randn(4, 8, 8, generator=torch.Generator("cpu").manual_seed(int(seed))) for seed in seeds])
        torch.testing.assert_close(guided, raw_to_lpg(noise - .3))
        unguided = prior.generate(2, labels, device="cpu", seeds=seeds, use_cfg=False)
        torch.testing.assert_close(unguided, raw_to_lpg(noise - .2))
        single = torch.cat([prior.generate(1, labels[i:i+1], device="cpu", seeds=seeds[i:i+1]) for i in range(2)])
        torch.testing.assert_close(guided, single, rtol=0, atol=0)

    def test_eval_passes_tensor_seeds_to_native_adapters(self):
        from eval_generation import build_models, parse_args
        from models.priors.registry import prior_defaults
        class Pipeline(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.diffusion = SimpleNamespace(generate=self.generate, defaults=prior_defaults("dmf"))
                self.tokenizer = nn.Identity()
            def generate(self, batch_size, **kwargs):
                self_seeds = kwargs["seeds"]
                self_seeds.cpu().tolist()  # Native DMF/MeanFlow contract.
                return torch.zeros(batch_size, 3, 8, 8)
            def decode(self, latents):
                return latents
        args = parse_args(["--prior_family", "dmf", "--out_dir", "unused"])
        with patch("models.model.LatentPriorGenerator", Pipeline):
            sample, metadata = build_models(args, "cpu")
            self.assertEqual(sample(torch.tensor([0]), torch.tensor([42])).shape, (1, 3, 8, 8))
            self.assertEqual(metadata["cfg_scale"], 1.)

    def test_titok_small_maskgit_preserves_per_sample_seeds(self):
        from models.titok.generator_arch import ImageBert
        from models.titok.generator import TiTokGenerator
        from models.titok.registry import config as titok_config
        cfg = titok_config("generator", "titok-L32")
        cfg.model.vq_model.codebook_size = 32
        gen = cfg.model.generator
        gen.hidden_size, gen.num_hidden_layers, gen.num_attention_heads = 32, 1, 4
        gen.intermediate_size, gen.image_seq_len = 64, 4
        wrapper = TiTokGenerator.__new__(TiTokGenerator)
        nn.Module.__init__(wrapper)
        wrapper.model = ImageBert(cfg).eval()
        wrapper.token_length = 4
        wrapper.cfg_omega = 4.5
        wrapper.sampling = dict(guidance_scale=4.5, guidance_decay="linear", guidance_scale_pow=3., randomize_temperature=9.5, softmax_temperature_annealing=True, num_sample_steps=8)
        seeds, labels = torch.tensor([42, 43]), torch.tensor([0, 1])
        full = wrapper.standard_generation(2, labels, device="cpu", seeds=seeds)
        single = torch.cat([wrapper.standard_generation(1, labels[i:i+1], device="cpu", seeds=seeds[i:i+1]) for i in range(2)])
        self.assertTrue(torch.equal(full, single))
        self.assertTrue(torch.all((full >= 0) & (full < 32)))


if __name__ == "__main__":
    unittest.main()
