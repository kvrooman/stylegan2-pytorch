
import os
import argparse
import pickle

import lpips
import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from scipy import linalg
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model_inception import InceptionV3
from model_stylegan import Generator
from run_prepare_data import MultiResolutionDataset


class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def calc_fid(sample_features, real_mean, real_cov, eps=1e-6):

    sample_mean = np.mean(sample_features, 0)
    sample_cov = np.cov(sample_features, rowvar=False)
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace
    return fid


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Calculate FID scores")
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument("--truncation_mean", type=int, default=4096, help="number of samples to calculate mean for truncation",)
    parser.add_argument("--batch", type=int, default=64, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=50000, help="number of the samples for calculating FID",)
    parser.add_argument("--size", type=int, default=256, help="image sizes for generator")
    parser.add_argument("--flip", action="store_true", help="apply random flipping to real images")
    parser.add_argument("--space", choices=["z", "w"], help="space that PPL calculated with")
    parser.add_argument("--eps", type=float, default=1e-4, help="epsilon for numerical stability")
    parser.add_argument("--crop", action="store_true", help="apply center crop to the images")
    parser.add_argument("--sampling", default="end", choices=["end", "full"], help="set endpoint sampling method")
    parser.add_argument("--lmdb_path", metavar="PATH", required=True, help="path to datset lmdb file")
    parser.add_argument("--ckpt", metavar="CHECKPOINT", required=True, help="path to generator checkpoint")
    parser.add_argument("--inception", type=str, default=None, required=True, help="path to precomputed inception embedding",)
    args = parser.parse_args()

    latent_dim = 512

    
    ckpt = torch.load(args.ckpt)
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(ckpt["g_ema"])
    g_ema = nn.DataParallel(g_ema).eval().to(device)

    inception = InceptionV3([3], normalize_input=False)
    inception = nn.DataParallel(inception).eval().to(device)
    
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

    # Inception Features of Generated Images
    with torch.no_grad():
        mean_latent = g_ema.mean_latent(args.truncation_mean) if args.truncation < 1 else None

        n_batch = args.n_sample // args.batch
        resid = args.n_sample - (n_batch * args.batch)
        batch_sizes = [args.batch] * n_batch + [resid]

        feature_list = []
        for batch in tqdm(batch_sizes):
            latent = torch.randn(batch, 512, device=device)
            img, _ = g_ema([latent], truncation=args.truncation, truncation_latent=mean_latent)
            feature = inception(img)[0].view(img.shape[0], -1)
            feature_list.append(feature.to("cpu"))

        sample_features = torch.cat(features, dim=0).numpy()
    print(f"extracted {sample_features.shape} features shape")

    # Inception Features of FFHQ / LMDB Images
    if not args.inception and args.lmdb_path:
        name = os.path.splitext(os.path.basename(args.lmdb_path))[0]
        transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
        dataset = MultiResolutionDataset(args.path, transform=transform, resolution=args.size)
        loader = DataLoader(dataset, batch_size=args.batch, num_workers=4)

        with torch.no_grad():
            feature_list = []
            for img in tqdm(loader):
                img = img.to(device)
                feature = inception(img)[0].view(img.shape[0], -1)
                feature_list.append(feature.to("cpu"))

            dataset_features = torch.cat(feature_list, 0).numpy()
        print(f"extracted lmdb {dataset_features.shape} features shape")

        dataset_features = dataset_features[: args.n_sample]
        real_mean = np.mean(dataset_features, 0)
        real_cov = np.cov(dataset_features, rowvar=False)

        with open(f"inception_{name}.pkl", "wb") as f:
            pickle.dump({"mean": real_mean, "cov": real_cov, "size": args.size, "path": args.path}, f)

    else:
        with open(args.inception, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]

    # Calc FID
    fid = calc_fid(sample_features, real_mean, real_cov)
    print("fid:", fid)

    # Calc PPL
    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]

    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            noise = g_ema.make_noise()

            inputs = torch.randn([batch * 2, latent_dim], device=device)
            if args.sampling == "full":
                lerp_t = torch.rand(batch, device=device)
            else:
                lerp_t = torch.zeros(batch, device=device)

            if args.space == "w":
                latent = g_ema.get_latent(inputs)
                latent_t0, latent_t1 = latent[::2], latent[1::2]
                latent_e0 = latent_t0 + (latent_t1 - latent_t0) * lerp_t[:, None]
                latent_e1 = latent_t0 + (latent_t1 - latent_t0) * (lerp_t[:, None] + args.eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            image, _ = g_ema([latent_e], input_is_latent=True, noise=noise)

            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (args.eps ** 2)
            distances.append(dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)

    print("ppl:", filtered_dist.mean())
