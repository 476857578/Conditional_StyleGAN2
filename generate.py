import csv
import torch
from model import Generator
import os
from torchvision import utils
from tqdm.contrib import tzip
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Generate_bubble_images(bubble_base, bubble_prompts, bubble_outputs):
    with open(bubble_base, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [list(map(float, row[1:])) for row in reader]
    AT = torch.tensor(data).to(device)          # output:(329, 8)
    pseudo_inverse = torch.pinverse(AT.T).T.to(device)       # (329, 8)➡️(8, 329)➡️(329, 8)➡️(8, 329)

    with open(bubble_prompts, 'r') as f:
        filenames = []
        bubble_data = []
        for line in f:
            parts = line.strip().split('\t')
            filenames.append(parts[0])
            bubble_data.append(parts[1:])
    bubble_tensor = torch.tensor([[float(value) for value in row] for row in bubble_data], device=device)   # output:[bs,8]

    generated_bubble_weights = torch.matmul(bubble_tensor, pseudo_inverse)     # input:[bs,8][8,329]  output:[bs,329]

    generate_StyleGAN = Generator(128, 329, 6, channel_divide=1).to(device)
    generate_StyleGAN.eval()
    checkpoint = torch.load('/home/yubd/mount/codebase/Conditional_StyleGAN2/checkpoint/100000.pt', map_location=device, weights_only=False)
    generate_StyleGAN.load_state_dict(checkpoint["g_ema"])

    sample_z = torch.randn(len(filenames), 329, device=device)
    noises_file = os.path.join(bubble_outputs, 'noises_file.txt')

    with open(noises_file, 'w') as f:
        for z in sample_z:
            f.write('\t'.join(map(str, z.tolist())) + '\n')

    with torch.no_grad():
        samples, _ = generate_StyleGAN([sample_z], generated_bubble_weights)
        for sample, filename in tzip(samples, filenames):
            utils.save_image(
                1 - sample,
                os.path.join(bubble_outputs, filename),
                nrow=1,
                normalize=True,
                value_range=(0, 1),
                pad_value=1,
                padding=0,
                )
            
        utils.save_image(1 - samples, os.path.join(bubble_outputs, 'image_grid.png'), 
                        nrow=8,
                        pad_value=1,
                        padding=0,
                )

if __name__ == "__main__":
    bubble_base = '/home/yubd/mount/codebase/Conditional_StyleGAN2/bubble_test/path_to_base_csv_file.csv'
    bubble_prompts = '/home/yubd/mount/codebase/Conditional_StyleGAN2/bubble_test/64bubbles/bubble_prompts.txt'
    bubble_outputs = '/home/yubd/mount/codebase/Conditional_StyleGAN2/bubble_test/64bubbles/output'
    if not os.path.exists(bubble_outputs):
        os.makedirs(bubble_outputs)
    Generate_bubble_images(bubble_base, bubble_prompts, bubble_outputs)
