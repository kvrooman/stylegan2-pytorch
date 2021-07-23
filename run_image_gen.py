import argparse

import torch
from tqdm import tqdm
from torchvision import utils

from model_stylegan import Generator


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Apply closed form factorization")
    parser.add_argument("--index", type=int, default=0, help="index of eigenvector")
    parser.add_argument("--degree", type=float, default=5, help="scalar factors for moving latent vectors along eigenvector",)
    parser.add_argument("--channel_multiplier", type=int, default=2, help='channel multiplier factor. config-f = 2, else = 1',)
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument("--size", type=int, default=256, help="output image size of the generator")
    parser.add_argument("--n_sample", type=int, default=7, help="number of samples created in each image")
    parser.add_argument("--pics", type=int, default=10, help="seperate number of images to be generated")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation factor")
    parser.add_argument("--truncation_mean", type=int, default=4096, help="number of vectors to calculate mean for the truncation")
    parser.add_argument("--show_factors", action="store_true", help="apply non leaking augmentation")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    modulate = {k: v
                for k, v in ckpt["g_ema"].items()
                if "modulation" in k and "to_rgbs" not in k and "weight" in k}

    weight_mat = list(modulate.values())
    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V
    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)

    checkpoint = torch.load(args.ckpt)
    g_ema = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier)
    g_ema.load_state_dict(checkpoint["g_ema"], strict=False).eval().to(device)

    for pic_num in tqdm(range(args.pics)):
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean) if args.truncation < 1 else None
            sample_z = torch.randn(args.n_sample*args.n_sample, args.latent, device=device)
            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)
            utils.save_image(sample,
                             f"data/{str(pic_num).zfill(6)}.png",
                             nrow=args.n_sample,
                             normalize=True,
                             range=(-1, 1))

            if args.show_factors:
                eigvec = torch.load(args.factor)["eigvec"].to(device)
                direction = args.degree * eigvec[:, args.index].unsqueeze(0)
                latent = g_ema.get_latent(torch.randn(args.n_sample, 512, device=device))

                img, _ = g_ema([latent], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True)
                img1, _ = g_ema([latent + direction], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True)
                img2, _ = g_ema([latent - direction], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True)
                combined_image = torch.cat([img1, img, img2], 0)
                utils.save_image(combined_image,
                                 f"data/factor_index-{args.index}_degree-{args.degree}.png",
                                 normalize=True,
                                 range=(-1, 1),
                                 nrow=args.n_sample)
