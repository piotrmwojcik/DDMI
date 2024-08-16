"""Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
"""
import copy
import os
# Enable import from parent package
import sys
from functools import partial

import torch
import trimesh
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage

import wandb

from models.siren import Siren

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
import numpy as np
from torch.utils.data import DataLoader

def get_model(cfg):
    if cfg.model_type == "siren":
        model = Siren(**cfg.mlp_config)
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print("Total number of parameters: %d" % nparameters)

    return model


def get_image_tensor(image_path, sidelength):
    # Load the image from the specified file path (ensure it's in RGB mode)
    img = Image.open(image_path).convert('RGB')

    # Define the transformation pipeline
    transform = Compose([
        Resize((sidelength, sidelength)),  # Resize to the specified side length
        ToTensor(),                        # Convert the image to a tensor
        #Normalize(mean=torch.Tensor([0.5, 0.5, 0.5]), std=torch.Tensor([0.5, 0.5, 0.5]))  # Normalize the tensor
    ])

    # Apply the transformations
    img = transform(img)
    return img


class ImageFitting(Dataset):
    def __init__(self, sidelength, path):
        super().__init__()
        img = get_image_tensor(path, sidelength)
        self.pixels = img.permute(1, 2, 0).view(sidelength*sidelength, 3)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        print(self.coords.shape, self.pixels.shape)
        return self.coords, self.pixels

def main(cfg: DictConfig):
    wandb.init(
        project="hyperdiffusion_overfitting",
        dir=cfg.wandb_dir,
        config=dict(cfg),
        mode="online",
    )
    first_state_dict = None
    if cfg.strategy == "same_init":
        first_state_dict = get_model(cfg).state_dict()
    x_0s = []
    curr_lr = cfg.lr
    root_path = os.path.join(cfg.logging_root, cfg.exp_name)
    multip_cfg = cfg.multi_process
    files = [
        file
        for file in os.listdir(cfg.dataset_folder)
    ]
    if multip_cfg.enabled:
        if multip_cfg.ignore_first:
            files = files[1:]  # Ignoring the first one
        count = len(files)
        per_proc_count = count // multip_cfg.n_of_parts
        start_index = multip_cfg.part_id * per_proc_count
        end_index = min(count, start_index + per_proc_count)
        files = files[start_index:end_index]
    lengths = []
    names = []

    for i, file in enumerate(files):

        filename = file.split(".")[0]
        filename = f"{filename}_jitter_{j}"

        sdf_dataset = ImageFitting(
            path=os.path.join(cfg.dataset_folder, file),
            sidelength=128)
        dataloader = DataLoader(
            sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0
        )
        if cfg.strategy == "save_pc":
            continue
        elif cfg.strategy == "diagnose":
            lengths.append(len(sdf_dataset.coords))
            names.append(file)
            continue

        # Define the model.
        model = get_model(cfg).cuda()

        # Define the loss
        loss_fn = loss_functions.sdf
        if cfg.output_type == "occ":
            loss_fn = (
                loss_functions.occ_tanh
                if cfg.out_act == "tanh"
                else loss_functions.occ_sigmoid
            )
        loss_fn = partial(loss_fn, cfg=cfg)
        #summary_fn = utils.wandb_sdf_summary

        filename = f"{cfg.output_type}_{filename}"
        checkpoint_path = os.path.join(root_path, f"{filename}_model_final.pth")
        if os.path.exists(checkpoint_path):
            print("Checkpoint exists:", checkpoint_path)
            continue
        if cfg.strategy == "continue":
            if not os.path.exists(checkpoint_path):
                continue
            model.load_state_dict(torch.load(checkpoint_path))
        elif (
            first_state_dict is not None
            and cfg.strategy != "random"
            and cfg.strategy != "first_weights_kl"
        ):
            print("loaded")
            model.load_state_dict(first_state_dict)

        training.train(
            model=model,
            train_dataloader=dataloader,
            epochs=cfg.epochs,
            lr=curr_lr,
            steps_til_summary=cfg.steps_til_summary,
            epochs_til_checkpoint=cfg.epochs_til_ckpt,
            model_dir=root_path,
            loss_fn=loss_fn,
            summary_fn=None,
            double_precision=False,
            clip_grad=cfg.clip_grad,
            wandb=wandb,
            filename=filename,
            cfg=cfg,
        )
        if (
            i == 0
            and first_state_dict is None
            and (
                cfg.strategy == "first_weights"
                or cfg.strategy == "first_weights_kl"
            )
            and not multip_cfg.enabled
        ):
            first_state_dict = model.state_dict()
            print(curr_lr)
        state_dict = model.state_dict()

        # Calculate statistics on the MLP
        weights = []
        for weight in state_dict:
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights)
        x_0s.append(weights)
        tmp = torch.stack(x_0s)
        var = torch.var(tmp, dim=0)
        print(
            var.shape,
            var.mean().item(),
            var.std().item(),
            var.min().item(),
            var.max().item(),
        )
        print(var.shape, torch.var(tmp))


if __name__ == "__main__":
    main()
