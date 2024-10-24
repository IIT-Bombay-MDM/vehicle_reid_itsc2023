import argparse
import json
import os
import random
import shutil
from typing import OrderedDict

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.triplet_sampler import CustomDataSet4Veri776_withviewpont
from metrics.eval_reid import eval_func
from processor import get_model
from utils import re_ranking


def normalize_batch(batch, maximo=None, minimo=None):
    if maximo is not None:
        return (batch - minimo.unsqueeze(-1).unsqueeze(-1)) / (
            maximo.unsqueeze(-1).unsqueeze(-1) - minimo.unsqueeze(-1).unsqueeze(-1)
        )
    else:
        return (batch - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) / (
            torch.amax(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
        )


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_single_image(
    model,
    device,
    query_info,
    dataloader_g,
    remove_junk=True,
    scaler=None,
    re_rank=False,
    output_dir="./results",
):
    model.eval()
    qf = []
    gf = []
    g_camids = []
    g_vids = []

    # Create a unique directory for the current test inside the results folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng()
    # rejection sampling
    while True:
        test_dir = os.path.join(
            output_dir, "test_" + str(rng.integers(1000, 9999))
        )  # adjust this based on req
        if not os.path.isdir(test_dir):
            break
    os.makedirs(test_dir)

    # Inference on the query image
    with torch.no_grad():
        (query_image, q_id, cam_id, view_id), query_image_path = query_info
        query_image = query_image.unsqueeze(0).to(device)
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, _, ffs, _ = model(query_image, cam_id, view_id)
        else:
            _, _, ffs, _ = model(query_image, cam_id, view_id)

        end_vec = []
        for item in ffs:
            end_vec.append(F.normalize(item))
        qf.append(torch.cat(end_vec, 1))

    del query_image

    shutil.copy(query_image_path, os.path.join(test_dir, "query_image.jpg"))

    # Inference on gallery images
    image_paths = {}
    sampler_indices = list(data_g.sampler)
    with torch.no_grad():
        count_imgs = 0
        for batch_idx, (image, g_id, cam_id, view_id) in enumerate(
            tqdm(
                dataloader_g,
                desc="Gallery infer (%)",
                bar_format="{l_bar}{bar:20}{r_bar}",
            )
        ):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                _, _, ffs, _ = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            gf.append(torch.cat(end_vec, 1))
            g_vids.append(g_id)
            g_camids.append(cam_id)

            start_idx = batch_idx * data_g.batch_size
            # Account for the last batch, which may be smaller
            end_idx = start_idx + image.size(0)

            # Get the dataset indices for this batch
            current_indices = sampler_indices[start_idx:end_idx]

            for dataset_idx in current_indices:
                image_paths[dataset_idx] = data_g.dataset.get_image_path(dataset_idx)

            count_imgs += image.shape[0]

    qf = torch.cat(qf, dim=0)
    gf = torch.cat(gf, dim=0)

    # Calculate distance matrix
    m, n = qf.shape[0], gf.shape[0]
    if re_rank:
        distmat = re_ranking(qf, gf, k1=80, k2=16, lambda_value=0.3)
    else:
        distmat = (
            torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        distmat = torch.sqrt(distmat).cpu().numpy()

    g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
    g_vids = torch.cat(g_vids, dim=0).cpu().numpy()

    cmc, mAP = eval_func(
        distmat,
        np.array([q_id]),
        g_vids,
        np.array([cam_id]),
        g_camids,
        remove_junk=remove_junk,
    )
    print(f"mAP = {mAP},  CMC1= {cmc[0]}, CMC5= {cmc[4]}")

    # Save top matches
    sorted_indices = np.argsort(distmat[0])  # Sort by ascending distance
    top_matches_dir = os.path.join(test_dir, "top_matches")
    os.makedirs(top_matches_dir, exist_ok=True)
    matches_dist = {}
    for rank, idx in enumerate(
        sorted_indices[:30]
    ):  # Save top 30 matches, since some are very similar
        matches_dist[f"rank_{rank + 1}"] = float(distmat[0][idx])
        match_image_path = image_paths[idx]
        match_save_path = os.path.join(top_matches_dir, f"rank_{rank + 1}.jpg")
        shutil.copy(match_image_path, match_save_path)

    with open(os.path.join(test_dir, "distances.json"), "w") as f:
        json.dump(matches_dist, f, indent=4)

    print(f"matches saved in {test_dir}")

    return cmc, mAP


if __name__ == "__main__":
    set_seed(0)
    parser = argparse.ArgumentParser(description="Reid train")

    parser.add_argument(
        "--batch_size", default=None, type=int, help="an integer for the accumulator"
    )
    parser.add_argument(
        "--dataset", default=None, help="Choose one of[Veri776, VERIWILD]"
    )
    parser.add_argument("--model_arch", default=None, help="Model Architecture")
    parser.add_argument(
        "--path_weights", default=None, help="Path to *.pth/*.pt loading weights file"
    )
    parser.add_argument("--re_rank", action="store_true", help="Re-Rank")
    parser.add_argument(
        "--query_image_id", default=None, help="Path to the single query image"
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory to save the query and matched results",
    )
    args = parser.parse_args()

    with open(args.path_weights + "config.yaml", "r") as stream:
        data = yaml.safe_load(stream)

    data["BATCH_SIZE"] = args.batch_size or data["BATCH_SIZE"]
    data["dataset"] = args.dataset or data["dataset"]
    data["model_arch"] = args.model_arch or data["model_arch"]

    teste_transform = transforms.Compose(
        [
            transforms.Resize((data["y_length"], data["x_length"]), antialias=True),
            transforms.Normalize(data["n_mean"], data["n_std"]),
        ]
    )

    if data["half_precision"]:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = False

    if data["dataset"] == "Veri776":
        data_q = CustomDataSet4Veri776_withviewpont(
            data["query_list_file"],
            data["query_dir"],
            data["train_keypoint"],
            data["test_keypoint"],
            is_train=False,
            transform=teste_transform,
        )
        data_g = CustomDataSet4Veri776_withviewpont(
            data["gallery_list_file"],
            data["teste_dir"],
            data["train_keypoint"],
            data["test_keypoint"],
            is_train=False,
            transform=teste_transform,
        )
        data_g = DataLoader(
            data_g,
            batch_size=data["BATCH_SIZE"],
            shuffle=False,
            num_workers=data["num_workers_teste"],
        )

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Selected device: {device}")

    model = get_model(data, torch.device("cpu"))

    # One of the saved weights last.pt best_CMC.pt best_mAP.pt
    path_weights = args.path_weights + "best_mAP.pt"

    try:
        model.load_state_dict(torch.load(path_weights, map_location="cpu"))
    except RuntimeError:
        tmp = torch.load(path_weights, map_location="cpu")
        tmp = OrderedDict((k.replace("module.", ""), v) for k, v in tmp.items())
        model.load_state_dict(tmp)

    model = model.to(device)
    model.eval()

    if args.query_image_id:
        idx = int(args.query_image_id)
        cmc, mAP = test_single_image(
            model,
            device,
            (data_q[idx], data_q.get_image_path(idx)),
            data_g,
            remove_junk=False,  # todo: remove_junk=True is currently broken
            scaler=scaler,
            re_rank=args.re_rank,
            output_dir=args.output_dir,
        )
    else:
        print("Please provide a query image id using --query_image_id.")
