import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
import tqdm
import torchvision.transforms.functional as TF
import itertools
VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training
"""
IMPORT YOUR MODEL HERE
"""
from train.models.gnm.gnm import GNM
from train.models.vint.vint import ViNT
from train.models.EffoNav.vint_dino import ViNT_DINO

from train.models.vint.vit import ViT
from train.models.nomad.nomad import NoMaD, DenseNetwork
from train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

from train.models.nomad_dino.nomad_dino import NoMaD_DINO, DenseNetwork_DINO
from train.models.nomad_dino.nomad_vint_dino import NoMaD_ViNT_DINO#, replace_bn_with_gn
from train.models.nomad_dino.utilities import DinoV2ExtractFeatures
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_dino.train.train.data.nav_dataset import ViNT_Dataset
from train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)
from train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from train.visualizing.visualize_utils import to_numpy, from_numpy

import torch.nn.functional as F

def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset


    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        DINO=None

    elif config["model_type"] == "vint_dino":
        model = ViNT_DINO(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        DINO=DinoV2ExtractFeatures(dino_model="dinov2_vits14", layer=11, facet='value',device="cuda")

    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            # print("replace_bn_with_gn")
            vision_encoder = replace_bn_with_gn(vision_encoder)
            # print("replace_bn_with_gn end")

        elif config["vision_encoder"] == "vib": 
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        DINO=None
    else:
        raise ValueError(f"Model {config['model']} not supported")



    current_epoch = 0

    weight_path = os.path.join("model_weights", config["model_type"]+".pth")
    print("Loading model from ", weight_path)
    latest_checkpoint = torch.load(weight_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
    load_model(model, config["model_type"], latest_checkpoint)
    model=model.to(device)

    action_loss_dict={}
    for dataset_type in test_dataloaders:
        print(
            f"Start {dataset_type} ViNT Testing "
        )
        loader = test_dataloaders[dataset_type]
        num_batches = len(loader)
        num_batches = max(int(num_batches * config['eval_fraction']), 1)
        with torch.no_grad():
            tqdm_iter = tqdm.tqdm(
                itertools.islice(loader, num_batches),
                total=num_batches,
                disable=not True,
                dynamic_ncols=True,
                desc=f"Evaluating {dataset_type}",
            )
            idx=0
            loss=0
            for i, data in enumerate(tqdm_iter):
                (
                    obs_image,
                    goal_image,
                    action_label,
                    dist_label,
                    goal_pos,
                    dataset_index,
                    action_mask,
                ) = data

                obs_images = torch.split(obs_image, 3, dim=1)
                viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
                obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

                goal_image = transform(goal_image).to(device)
                if config["model_type"] == "vint_dino":
                    model_outputs = model(obs_image, goal_image, DINO)
                else:
                    model_outputs = model(obs_image, goal_image)

                dist_label = dist_label.to(device)
                action_label = action_label.to(device)
                action_mask = action_mask.to(device)

                dist_pred, action_pred = model_outputs

                def action_reduce(unreduced_loss: torch.Tensor):
                    # Reduce over non-batch dimensions to get loss per batch element
                    while unreduced_loss.dim() > 1:
                        unreduced_loss = unreduced_loss.mean(dim=-1)
                    assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                    return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

                # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
                assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
                action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

                # action_loss=torch.nn.MSELoss()(action_label,action_pred)
                loss+=action_loss.item()
                # print(action_loss.item())
                # visualize_traj_pred(
                #     to_numpy(viz_obs_image),
                #     to_numpy(viz_goal_image),
                #     to_numpy(dataset_index),
                #     to_numpy(goal_pos),
                #     to_numpy(action_pred),
                #     to_numpy(action_label),
                #     dataset_type,
                #     config["normalize"],
                #     os.path.join("logs",config["model_type"]),
                #     0,
                #     config["num_images_log"],
                # )
                idx+=1
                if idx>=20:
                    print(dataset_type,"action loss:",loss/idx)
                    action_loss_dict[dataset_type]=loss/idx
                    break

    print(action_loss_dict)



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint_dino.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    print(config)
    main(config)
