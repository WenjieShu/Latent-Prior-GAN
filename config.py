


### require major revision
def parse_args():
    parser = argparse.ArgumentParser(description="Latent Prior GAN on iMF models.")
    parser.add_argument("--config", default="", type=str, help="optional YAML config")
    parser.add_argument("--dataset_dir", required=False, default="/mnt/nfs/wenjie/dataset/imagenet_idx")
    parser.add_argument("--dataset_name", default="ImageNet", choices=["ImageNet"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--model_type", default="iMF-XL-2", choices=["iMF-B-2", "iMF-M-2", "iMF-L-2", "iMF-XL-2"])
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--vae_type", default="mse", choices=["mse", "ema"])
    parser.add_argument("--pretrained_imf_pytorch", required=False, default="")
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--use_cfg", action="store_true", default=False)
    parser.add_argument("--cfg_omega", type=float, default=None)
    parser.add_argument("--cfg_t_min", type=float, default=None)
    parser.add_argument("--cfg_t_max", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--disc_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume", default="", type=str)

    parser.add_argument("--disc_weight", type=float, default=0.7)
    parser.add_argument("--lecam_loss_weight", type=float, default=0.001)
    parser.add_argument("--disc_cr_loss_weight", type=float, default=4.0)

    parser.add_argument("--checkpoint_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/checkpoints/extension_gar")
    parser.add_argument("--results_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/results/extension_gar")
    parser.add_argument("--yaml_dir", default="/mnt/nfs/wenjie/IMF-GAP_V2/yaml/extension_gar")
    parser.add_argument("--saver_name_pre", default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="IMF-GAP-V2")
    parser.add_argument("--wandb_run_name", default="")

    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--local-rank", default=-1, type=int)

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown arguments: {unknown}")

    config = load_config_file(args.config)
    if config:
        defaults = {k: v for k, v in config.items() if hasattr(args, k)}
        parser.set_defaults(**defaults)
        args = parser.parse_args()

    if not args.pretrained_imf_pytorch:
        args.pretrained_imf_pytorch = (
            f"/mnt/nfs/wenjie/IMF-GAP_V1/checkpoints/pytorch/{args.model_type}.pt"
        )
    if not args.saver_name_pre:
        cfg_tag = "cfg" if args.use_cfg else "nocfg"
        args.saver_name_pre = f"extension_gar_{args.model_type}_nt1_gan_only_{cfg_tag}"


    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return args

