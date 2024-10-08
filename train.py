# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
from medmnist.dataset import RetinaMNIST

from TNBC_dataset import TNBCDataset
from download import find_model

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics.image.kid import KernelInceptionDistance

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
import wandb
from torchvision.utils import make_grid

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir, cache=False):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.cache_dict = {}
        self.cache = cache
        if cache:
            for idx in range(len(self.features_files)):
                feature_file = self.features_files[idx]
                label_file = self.labels_files[idx]
                features = np.load(os.path.join(self.features_dir, feature_file))
                labels = np.load(os.path.join(self.labels_dir, label_file))
                self.cache_dict[idx] = (features, labels)

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        if self.cache:
            if idx in self.cache_dict:
                features, labels = self.cache_dict[idx]
            else:
                raise Exception('Cache no initalized properly')

        else:
            feature_file = self.features_files[idx]
            label_file = self.labels_files[idx]

            features = np.load(os.path.join(self.features_dir, feature_file))
            labels = np.load(os.path.join(self.labels_dir, label_file))

        return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name='Fast-DiT',
        config=vars(args))

    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.fold}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
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
        in_channels=args.in_channels
    )

    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    if os.path.exists(args.load_from):
        state_dict = find_model(os.path.join(args.load_from, 'best_ckpt.pt'))

        model.load_state_dict(state_dict)
    else:
        print('Initalizing Model randomly')
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    if not args.image_space:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    kid = KernelInceptionDistance(subset_size=args.val_samples // 4).to(device)

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    if 'MNIST' in args.data_path:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ])

    if 'imagenet' in args.data_path:
        val_dataset = ImageFolder(args.data_path, transform=transform)
    elif 'RetinaMNIST' in args.data_path:
        val_dataset = RetinaMNIST(root=args.data_path, as_rgb=True, transform=transform, size=224,
                                  download=True,
                                  split='val',
                                  target_transform=transforms.Compose([
                                      lambda x: torch.LongTensor(x),  # or just torch.tensor
                                      lambda x: F.one_hot(x, args.num_classes)])
                                  )

    elif 'TNBC' in args.data_path or 'oct' in args.data_path:
        val_dataset = TNBCDataset(args.data_path, transform=transform, mode='val', fold=args.fold)

    val_dl = DataLoader(val_dataset, batch_size=int(args.val_samples // accelerator.num_processes),
                        num_workers=args.num_workers,
                        sampler=None, drop_last=True, shuffle=True, pin_memory=False)

    val_dl = accelerator.prepare(val_dl)
    real_samples = next(iter(val_dl))[0]
    real_samples = (real_samples + 1.0) * 0.5
    real_samples = (real_samples * 255).type(torch.uint8).to(device)

    # Setup data:

    if not args.image_space:
        features_dir = f"{args.feature_path}/features_{args.fold}"
        labels_dir = f"{args.feature_path}/labels_{args.fold}"
        dataset = CustomDataset(features_dir, labels_dir, cache=args.cache)
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ])

        if 'imagenet' in args.data_path:
            dataset = ImageFolder(args.data_path, transform=train_transform)
        elif 'RetinaMNIST' in args.data_path:
            train_transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            ])

            dataset = RetinaMNIST(root=args.data_path, as_rgb=True, transform=train_transform, size=224,
                                  download=True,
                                  split='train',
                                  target_transform=transforms.Compose([
                                      lambda x: torch.LongTensor(x),  # or just torch.tensor
                                      lambda x: F.one_hot(x, args.num_classes)])
                                  )
        elif 'TNBC' in args.data_path or 'oct' in args.data_path:
            dataset = TNBCDataset(args.data_path, transform=train_transform, mode='train', fold=args.fold,
                                  cache=args.cache)

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    in_channels = args.in_channels
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    best_kid = 1e6
    running_loss = 0
    running_loss_large = 0
    running_loss_medium = 0
    running_loss_tiny = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)  # B X 4

            y = y.float()

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, multi_scale=args.multi_scale)

            with accelerator.accumulate(model):

                loss_dict = diffusion.training_losses(model, x, t, model_kwargs, multi_scale=args.multi_scale,
                                                      NCA_model='GeCA' in args.model)
                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                accelerator.backward(loss)
                if args.grad_norm:
                    # TODO: don't normalize gradients for non-CA models
                    for name, p in model.named_parameters():
                        if p.grad is not None and p.requires_grad:
                            p.grad /= (p.grad.norm() + 1e-8)

                opt.step()
                update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            if args.multi_scale:
                running_loss_large += loss_dict["loss_0"].mean().item()
                running_loss_medium += loss_dict["loss_1"].mean().item()
                running_loss_tiny += loss_dict["loss_2"].mean().item()

            log_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                if args.multi_scale:
                    avg_loss_large = torch.tensor(running_loss_large / log_steps, device=device)
                    avg_loss_large = avg_loss_large.item() / accelerator.num_processes

                    avg_loss_medium = torch.tensor(running_loss_medium / log_steps, device=device)
                    avg_loss_medium = avg_loss_medium.item() / accelerator.num_processes

                    avg_loss_tiny = torch.tensor(running_loss_tiny / log_steps, device=device)
                    avg_loss_tiny = avg_loss_tiny.item() / accelerator.num_processes

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                    accelerator.log({'train_loss': avg_loss},
                                    step=train_steps)
                    if args.multi_scale:
                        accelerator.log({
                            'train_loss_large': avg_loss_large,
                            'train_loss_medium': avg_loss_medium,
                            'train_loss_tiny': avg_loss_tiny,
                        },

                            step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                running_loss_large = 0
                running_loss_medium = 0
                running_loss_tiny = 0

                log_steps = 0
                start_time = time()

            train_steps += 1
            # Save DiT checkpoint:
            # if train_steps % args.ckpt_every == 0 and train_steps > 0:

        if epoch % args.validate_every == 0:
            #
            model.eval()  # important! This disables randomized embedding dropout
            #

            with torch.no_grad():

                z = torch.randn(args.val_samples, in_channels, latent_size, latent_size, device=device)
                #         # Idle phase

                y = torch.zeros(size=(args.val_samples,
                                      args.num_classes)).to(device)
                # model_kwargs = dict(y=y)
                model_kwargs = dict(y=y, multi_scale=args.multi_scale)

                if 'GeCA' in args.model:
                    model_kwargs['extras'] = model.module.seed(z, [latent_size, latent_size])

                sample_fn = model.forward
                #
                #         # Sample images:
                samples = diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                #
                if not args.image_space:
                    samples = vae.decode(samples / 0.18215).sample
                    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 1, 2, 3).to(device,
                                                                                                  dtype=torch.uint8)
                else:
                    samples = (samples + 1.0) * 0.5
                    samples = (samples * 255).to(device, dtype=torch.uint8)

                kid.update(accelerator.gather_for_metrics(real_samples), real=True)
                kid.update(accelerator.gather_for_metrics(samples), real=False)
                kid_mean, kid_std = kid.compute()

                if accelerator.is_main_process:

                    accelerator.log({'KID mean': kid_mean},
                                    step=train_steps)
                    accelerator.log({'KID std': kid_std},
                                    step=train_steps)

                    if kid_mean < best_kid:
                        best_kid = kid_mean
                        if accelerator.is_main_process:
                            # TODO add module
                            checkpoint = {
                                "model": model.state_dict(),
                                "ema": ema.state_dict(),
                                "opt": opt.state_dict(),
                                "args": args
                            }
                            checkpoint_path = f"{checkpoint_dir}/best_ckpt.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")

                        imgs = make_grid(samples, nrow=8)
                        imgs = wandb.Image(imgs, caption='Hey')
                        accelerator.log({'examples': imgs},
                                        step=train_steps)

            model.train()  # important! This disables randomized embedding dropout

        # save_examples([gen_sample, real_sample],
        #               target_shape=np.array(train_ds.original_shape) // np.array([1, 4, 4]),
        #               log_path=LOG_PATH,
        #               epoch=epoch,
        #               names=['gen', 'real'],
        #               device=DEV)

    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")

    accelerator.end_training()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--data-path", type=str, default="data")

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--load_from", type=str, default="load_from")

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--val_samples", type=int, default=24)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=4)

    parser.add_argument("--grad_norm", action='store_true', default=False)

    parser.add_argument("--image_space", action='store_true', default=False)
    parser.add_argument("--cache", action='store_true', default=False)

    parser.add_argument("--multi_scale", action='store_true', default=False)

    args = parser.parse_args()

    main(args)
