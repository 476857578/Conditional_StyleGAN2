import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from dataloader import BubbleDataset
import pytorch_ssim
import csv
import pandas as pd
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

# from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


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
        # for batch in loader:
        #     yield batch
        for (real_img, real_label, fake_label, k3) in loader:
            yield real_img, real_label, fake_label, k3


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return (real_loss.mean() + fake_loss.mean()) / 2

def d_logistic_label_loss(real_pred, fake_pred1, fake_pred2, fake_pred3):
    real_loss = F.softplus(-real_pred)
    fake_loss1 = F.softplus(fake_pred1)
    fake_loss2 = F.softplus(fake_pred2)
    fake_loss3 = F.softplus(fake_pred3)
    return (real_loss.mean() + fake_loss1.mean() + fake_loss2.mean() + fake_loss3.mean()) / 4

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
    
    def forward(self, generated_images, real_images):
        batch_size = generated_images.size(0)
        
        # Calculate gradient maps for generated images
        generated_gradients = torch.abs(F.conv2d(generated_images, torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(generated_images.device), padding=1))
        
        # Set the first and last columns of generated_gradients to 0
        generated_gradients[:, :, :, 0] = 0
        generated_gradients[:, :, :, -1] = 0

        # Calculate pixel-wise mean squared error with gradient weights
        
        # Calculate pixel-wise mean squared error with gradient weights
        error_map = torch.pow(real_images - generated_images, 2) * generated_gradients

        # Check if error_map is all zeros
        if torch.all(error_map == 0):
            return 0
        
        non_zero_error_map = error_map[error_map != 0]
        error = torch.mean(non_zero_error_map)

        return error
    
def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    # list = r'/root/autodl-tmp/stylegan2_simple/parameters.txt'
    # list = r'/root/autodl-tmp/stylegan2_simple/parameters.txt'
    list = r'/root/autodl-tmp/stylegan2-pytorch-master/64bubble_test/struct_prompts.txt'
    # list = r'/root/autodl-tmp/struct_prompts_phi.txt'
    # 读取list文件
    with open(list, 'r') as f:
        lines = f.readlines()
    # 将数据转为float型
    k = [line.strip().split('\t')[1:] for line in lines]
    k = np.array(k, dtype=np.float32)
    # 转为tensor并放到device设备上
    k = torch.tensor(k).to(device)

    image_loss = torch.nn.MSELoss(reduction='mean')
    image_loss.cuda()

    edge_Loss = EdgeLoss().cuda()


    ssim_loss = pytorch_ssim.SSIM(window_size = 11)
    ssim_loss = ssim_loss.cuda()

    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break

        real_img, real_label, fake_label, k3 = next(loader)
        real_img = real_img.to(device)
        real_label = real_label.to(device)
        fake_label = fake_label.to(device)
        k3 = k3.to(device)

        # 训练discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        # fake_img, _ = generator(noise)
        fake_img, _ = generator(noise, fake_label)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        # real_pred = discriminator(real_img_aug, real_label)     # 希望是1
        # fake_pred1 = discriminator(fake_img, fake_label)        # 希望是0
        # fake_pred2 = discriminator(fake_img, k3)                # 希望是0
        # fake_pred3 = discriminator(real_img_aug, fake_label)    # 希望是0
        # d_loss = d_logistic_label_loss(real_pred, fake_pred1, fake_pred2, fake_pred3)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        # loss_dict["fake_score"] = (fake_pred1.mean() + fake_pred2.mean() + fake_pred3.mean()) / 3

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            # real_pred = discriminator(real_img_aug, real_label)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # 训练generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        # fake_img, _ = generator(noise, fake_label)
        # noise1 = mixing_noise(args.batch, args.latent, args.mixing, device)
        # fake_img1, _ = generator(noise1, fake_label)
        noise2 = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img2, _ = generator(noise2, real_label)
        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred2 = discriminator(fake_img2)
        # fake_pred = discriminator(fake_img1, fake_label)
        NONS_loss = g_nonsaturating_loss(fake_pred2)
        SSIM_loss = 1 - ssim_loss(fake_img2, real_img)
        EDGE_loss = edge_Loss(fake_img2, real_img)
        g_loss = (NONS_loss + SSIM_loss + EDGE_loss) / 3
        # g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            # fake_img, latents = generator(noise, return_latents=True)
            fake_img, latents = generator(noise, fake_label, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        time = "%s"%datetime.now()
        list = [time,i,d_loss.item(),g_loss.item(),NONS_loss.item(), SSIM_loss.item(), EDGE_loss.item(),path_loss.item()]
        data = pd.DataFrame([list])
        data.to_csv(args.loss_dir,mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss.item():.4f}; g: {g_loss.item():.4f}; path_loss: {path_loss:.4f}; "
                    f"NONS: {NONS_loss.item():.4f}; SSIM_: {SSIM_loss.item():.4f}; EDGE: {EDGE_loss.item():.4f}"
                )
            )
            # pbar.set_description(
            #     (
            #         f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
            #         f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
            #         f"augment: {ada_aug_p:.4f}"
            #     )
            # )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 500 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z], k)
                    utils.save_image(
                        1 - sample,
                        f"/root/autodl-tmp/stylegan2-pytorch-master/Ablation/sample5/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(0, 1),
                        pad_value = 1,
                        padding = 0,
                    )

            if i % 5000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"/root/autodl-tmp/stylegan2-pytorch-master/Ablation/sample5/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=200001, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=128, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=1,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        # default=None,
        default='/root/autodl-tmp/stylegan2-pytorch-master/checkpoint/100000.pt',
        # help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--loss_dir",
        type=str,
        # default=None,
        default="/root/autodl-tmp/stylegan2-pytorch-master/Ablation/sample5/Ablation5.csv",
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # args.latent = 256
    args.latent = 329
    args.n_mlp = 6
    args.para = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, args.para, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # # 读取csv文件
    # with open('/root/autodl-fs/A.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # 跳过表头
    #     data = [list(map(float, row[1:])) for row in reader]
    # QB_vector = np.array(data).T
    # QB_vector = torch.tensor(QB_vector, dtype=torch.float32, device=device, requires_grad=False)

    dataset = BubbleDataset(
        data_root = "/root/autodl-tmp/train",
        excel_dir = "/root/autodl-tmp/bubble_weights_nophi.csv",
    )

    loader = data.DataLoader(
        dataset,
        batch_size = args.batch,
        shuffle=True,
        )
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    # 记录loss 写为csv文件
    df = pd.DataFrame(columns=['time','iter','d_loss', 'g_loss', 'NONS_loss', 'SSIM_loss', 'EDGE_loss', 'path_loss'])#列名
    df.to_csv(args.loss_dir, index=False) #路径可以根据需要更改
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)