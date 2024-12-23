import argparse
import numpy as np
import csv

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
# import imageio
import os

def read_png_files(path):
    png_files = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            png_files.append(file)
    return png_files

def read_csv_file(file_path):
    bubble_params = {}
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            bubble_params[row[0]] = [float(val) for val in row[1:-2]]
    return bubble_params

def interpolate_params(params1, params2, params3, params4, num_steps):
    interpolated_params = []
    for i in range(num_steps):
        for j in range(num_steps):
            a = i / (num_steps - 1)
            b = j / (num_steps - 1)
            interpolated_param = []
            for k in range(len(params1)):
                param = (1 - a) * (1 - b) * params1[k] + a * (1 - b) * params2[k] + a * b * params3[k] + (1 - a) * b * params4[k]
                interpolated_param.append(param)
            interpolated_params.append(interpolated_param)
    return interpolated_params

def save_interpolated_params(interpolated_params, output_file):
    with open(output_file, 'w') as file:
        for params in interpolated_params:
            line = ' '.join([str(param) for param in params])
            file.write(line + '\n')

def generate(args, g_ema, device):
    output_dir = os.path.join(args.base_path, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    png_files = read_png_files(args.base_path)
    bubble_params = read_csv_file(args.weights_path)
    params1 = bubble_params.get(png_files[0])
    params2 = bubble_params.get(png_files[1])
    params3 = bubble_params.get(png_files[2])
    params4 = bubble_params.get(png_files[3])
    interpolated_params = interpolate_params(params1, params2, params3, params4, 8)
    save_interpolated_params(interpolated_params, os.path.join(output_dir, 'interpolated_params.txt'))
    interpolated_params = np.array(interpolated_params, dtype=np.float32)
    interpolated_params = torch.tensor(interpolated_params).to(device)

    sample_z = torch.randn(args.sample, args.latent, device=device)
    noises_file = os.path.join(output_dir, 'noises_file.txt')
    with open(noises_file, 'w') as f:
        for i in range(args.sample):
            f.write(f"{sample_z[i].tolist()}\n")

    with torch.no_grad():
        g_ema.eval()
        samples = []
        for i in tqdm(range(args.pics)):
            sample, _ = g_ema([sample_z], interpolated_params[i].unsqueeze(0))
            samples.append(1 - sample)
            utils.save_image(
                1 - sample,
                os.path.join(output_dir, f'{str(i).zfill(2)}.png'),
                nrow=1,
                normalize=True,
                value_range=(0, 1),
                pad_value=1,
                padding=0,
            )
        image_grid = torch.cat(samples, dim=0).reshape(-1, 1, 128, 128)
        utils.save_image(image_grid, os.path.join(output_dir, 'image_grid.png'), 
                        nrow=8,
                        pad_value=1,
                        padding=0,
                )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=128, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=64, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/yubd/mount/codebase/Conditional_StyleGAN2/checkpoint/100000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/yubd/mount/codebase/Conditional_StyleGAN2/bubble_test/fusion",
        help="path to the fusion",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="/home/yubd/mount/codebase/Conditional_StyleGAN2/bubble_test/bubble_weights.csv",
        help="path to the bubble_weights",
    )

    args = parser.parse_args()

    args.latent = 329
    args.n_mlp = 6

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_divide=1).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    g_ema.load_state_dict(checkpoint["g_ema"])

    generate(args, g_ema, device)