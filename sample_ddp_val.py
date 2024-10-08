# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
from pathlib import Path

from medmnist import RetinaMNIST
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.distributed as dist

from extract_features import center_crop_arr
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import pandas as pd

from torchvision import transforms
import torch.nn.functional as F

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


class CustomDataset(Dataset):
    def __init__(self, labels_dir, mode='train', fold=0, multi_class=False):

        self.datapath = Path(labels_dir)

        self.dataset_path = self.datapath / f'{mode}_{fold}.csv'
        if not self.dataset_path.exists():
            raise Exception(f"{self.dataset_path} could not be found")
        self.dataset = pd.read_csv(self.dataset_path, index_col=0)
        self.multi_class = multi_class
        # self.bin_column = bin_column
        self.label = self.get_mapped_labels(self.dataset, self.dataset_path)

    def get_mapped_labels(self, df, dataset_path):
        if self.multi_class:
            return list(df[self.bin_column].values)
        else:
            if 'oct' in str(dataset_path):
                return list(
                    df[['NOR', 'AMD', 'WAMD', 'DR', 'CSC', 'PED', 'MEM', 'FLD', 'EXU', 'CNV', 'RVO']].values)
            elif 'TNBC' in str(dataset_path):
                return list(df[['er', 'pr', 'her2', 'type_tn']].values)
            elif 'retina' in str(dataset_path):
                return list(df[['er', 'pr', 'her2', 'type_tn']].values)
            else:
                raise Exception('Error in get mapped labels')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label_file = self.label[idx]

        return 0, torch.from_numpy(label_file)


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    if args.image_space:
        # Create model:
        latent_size = args.image_size
    else:
        # Create model:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    in_channels = args.in_channels

    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    if not args.image_space:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"

    folder_name = f"{model_string_name}-GS-fold-{args.fold}-nstep-{args.num_sampling_steps}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(f'{sample_folder_dir}/gen_samples_val', exist_ok=True)

        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    # total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)

    if 'RetinaMNIST' in args.data_path:
        train_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ])
        dataset = RetinaMNIST(root=args.data_path, as_rgb=True, transform=train_transform, size=224,
                                  download=True,
                                  split='test',
                                  target_transform=transforms.Compose([
                                      lambda x: torch.LongTensor(x),  # or just torch.tensor
                                      lambda x: F.one_hot(x, 5)])
                                  )
        labels_name=['0','1','2','3','4']
    else:
        # Setup data:
        dataset = CustomDataset(args.data_path, mode='val', fold=args.fold)

        if 'oct' in args.data_path:
            labels_name = ['NOR', 'AMD', 'WAMD', 'DR', 'CSC', 'PED', 'MEM', 'FLD', 'EXU', 'CNV', 'RVO']
        elif 'TNBC' in args.data_path:
            labels_name = ['er', 'pr', 'her2', 'type_tn']

    loader = DataLoader(
        dataset,
        batch_size=int(args.per_proc_batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # if rank == 0:
    #     print(f"Total number of images that will be sampled: {total_samples}")
    # assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    # samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    # assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    # iterations = int(samples_needed_this_gpu // n)
    # pbar = range(iterations)
    # pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    annotations = None
    dict_pd = []

    def update_label(diction, labl):
        for name, label in zip(labels_name,
                               labl):
            diction[name] = int(label)

        return diction

    def get_new_labels(y):
        """ Convert each multilabel vector to a unique string """
        yy = [''.join(str(l)) for l in y]
        y_new = LabelEncoder().fit_transform(yy)
        return y_new

    for _ in range(args.expand_ratio):
        for _, labl in loader:

            labl = labl.to(device).float()
            labl = labl.squeeze(dim=1)  # B X 4

            z = torch.randn(labl.size(0), model.in_channels, latent_size, latent_size, device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.zeros_like(labl, device=device)
                y = torch.cat([labl, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=labl)
                sample_fn = model.forward

            if 'GeCA' in args.model:
                model_kwargs['extras'] = model.seed(z, [latent_size, latent_size])

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            if not args.image_space:
                samples = vae.decode(samples / 0.18215).sample
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
                                                                                              dtype=torch.uint8).numpy()
            else:
                samples = (samples + 1.0) * 0.5
                samples = (samples * 255).to(device, dtype=torch.uint8)

            # samples = vae.decode(samples / 0.18215).sample
            # samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
            #                                                                               dtype=torch.uint8).numpy()
            label = labl.cpu().numpy()
            annotations = label if annotations is None else np.concatenate([annotations, label], axis=0)

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                dict_client_img = {}
                index = i * dist.get_world_size() + rank + total
                dict_client_img['client_id'] = 'syn_' + str(index)
                dict_client_img['filename'] = os.path.join('gen_samples_val', f"{index:06d}.png")
                dict_client_img = update_label(dict_client_img, label[i])
                dict_client_img['patient_name'] = index
                dict_pd.append(dict_client_img)

                Image.fromarray(sample).save(f"{sample_folder_dir}/gen_samples_val/{index:06d}.png")
            total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        df = pd.DataFrame(dict_pd)
        multi_label_list = df[labels_name].values
        df['multi_class_label'] = get_new_labels(multi_label_list)
        df.to_csv(os.path.join(sample_folder_dir, f'val_syn_{args.fold}.csv'))

        np.save(os.path.join(f"{sample_folder_dir}/gen_samples_val_annotations.npz"), annotations)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--in_channels", type=int, default=4)

    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action='store_true', default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--expand_ratio", type=int, default=1)
    parser.add_argument("--image_space", action='store_true', default=False)


    args = parser.parse_args()
    main(args)
