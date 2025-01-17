
import os
import math
import random
import pickle
import argparse

import torch
import numpy as np
import torch.distributed as dist
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
    use_wandb = True
except ImportError:
    use_wandb = False

import model_custom_layers
from model_non_leaking import augment
from run_prepare_data import MultiResolutionDataset
from model_stylegan import Generator, Discriminator

def get_rank():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    return world_size

def synchronize():
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size() != 1:
            dist.barrier()

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor

def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with model_custom_layers.no_weight_gradients():
        grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()
    return path_penalty, path_mean.detach(), path_lengths

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        noises = torch.randn(batch, latent_dim, device=device)
    else:
        noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        mix = make_noise(batch, latent_dim, 2, device)
    else:
        mix = [make_noise(batch, latent_dim, 1, device)]
    return mix

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


class AdaptiveAugment:
    def __init__(self, ada_aug_target, ada_aug_len, update_every, device):
        self.r_t_stat = 0
        self.ada_aug_p = 0
        self.ada_update = 0
        self.ada_aug_len = ada_aug_len
        self.update_every = update_every
        self.ada_aug_target = ada_aug_target
        self.ada_aug_buf = torch.tensor([0.0, 0.0], device=device)

    @torch.no_grad()
    def tune(self, real_pred):
        adder = torch.tensor((torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=real_pred.device)
        self.ada_aug_buf += adder
        self.ada_update += 1

        if self.ada_update % self.update_every == 0:
            self.ada_aug_buf = reduce_sum(self.ada_aug_buf)
            pred_signs, n_pred = self.ada_aug_buf.tolist()
            self.r_t_stat = pred_signs / n_pred

            sign = 1 if self.r_t_stat > self.ada_aug_target else -1
            self.ada_aug_p += sign * n_pred / self.ada_aug_len
            self.ada_aug_p = min(1, max(0, self.ada_aug_p))
            self.ada_aug_buf.mul_(0)
            self.ada_update = 0

        return self.ada_aug_p


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    if get_rank() == 0:
        pbar = tqdm(range(args.iter), initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    else:
        pbar = range(args.iter, start=args.start_iter)

    r_t_stat = 0
    d_loss_val = 0
    g_loss_val = 0
    mean_path_length = 0
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    sample_z_for_images = [torch.randn(args.n_sample, args.latent, device=device)]

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    if args.use_adaptive_aug:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)


    loss_dict = {}
    for index in pbar:

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        real_img = next(loader).to(device)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.use_adaptive_aug:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = index % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = index % args.g_reg_every == 0
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, mean_path_length)

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()
            g_optim.step()
            mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()

        if get_rank() == 0:
            pbar.set_description((f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                                  f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                                  f"augment: {ada_aug_p:.4f}"))

            if args.wandb:
                real_score_val = loss_reduced["real_score"].mean().item()
                fake_score_val = loss_reduced["fake_score"].mean().item()
                path_length_val = loss_reduced["path_length"].mean().item()
                wandb.log({"Generator": g_loss_val,
                           "Discriminator": d_loss_val,
                           "Augment": ada_aug_p,
                           "Rt": r_t_stat,
                           "R1": r1_val,
                           "Path Length Regularization": path_loss_val,
                           "Mean Path Length": mean_path_length,
                           "Real Score": real_score_val,
                           "Fake Score": fake_score_val,
                           "Path Length": path_length_val})

            if index % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(sample_z_for_images)
                    utils.save_image(sample,
                                     f"sample/{str(index).zfill(6)}.png",
                                     nrow=int(args.n_sample ** 0.5),
                                     normalize=True,
                                     range=(-1, 1))

            if index % 10000 == 0:
                torch.save({"g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "args": args,
                            "ada_aug_p": ada_aug_p}, f"data/{str(index).zfill(6)}.pt")


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training",)
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization",)
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)",)
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization",)
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization",)
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training",)
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1",)
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation",)
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation",)
    args = parser.parse_args()

    args.n_mlp = 8
    args.latent = 512
    args.start_iter = 0
    args.distributed = if "WORLD_SIZE" in os.environ
    args.wandb = use_wandb and args.wandb
    args.use_adaptive_aug = args.augment and args.augment_p == 0

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(dataset,
                             batch_size=args.batch,
                             sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
                             drop_last=True)

    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).eval().to(device)
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),)
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),)

    if args.ckpt:
        print("loading model from:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(generator,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        broadcast_buffers=False)

        discriminator = nn.parallel.DistributedDataParallel(discriminator,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            broadcast_buffers=False)

    if get_rank() == 0 and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
