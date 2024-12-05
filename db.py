import argparse
import json
import os
from typing import List, Optional, OrderedDict, Union

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.triplet_sampler import CustomDataSet4Veri776_withviewpont
from processor import get_model

THRESHOLD = 0.7


class VehicleDB:
    """
    Represents a database of vehicle images.

    The vehicle images are uniquely identified by their:
        - vid
        - label
        - camid
    """

    def __init__(self, read_index: bool = False):
        if read_index:
            self.read_index()
        else:
            self.id_map = {}
            self.faiss_to_image_ids = {}
            self.index = None

    def add(
        self,
        label: int,
        camid: int,
        path: str,
        embs,
        vid: Optional[int] = None,
        transform_embs: bool = True,
    ) -> Optional[Tensor]:
        """
        Adds the image to the database.

        If an image with the same `vid`, `label` and `camid` exists,
        updates the image.

        Returns the already existing image, if any.
        """

        if not self.index and not vid:
            vid = 0  # first embedding

        embs_np = self.transform_embeddings(embs) if transform_embs else embs

        # create index if not already created
        if not self.index:
            dim = embs_np.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        if not vid:
            vid = self.search(embs_np, transform=False)
            if not vid:
                vid = max(self.id_map.keys(), default=0) + 1

        self.index.add(embs_np)
        faiss_id = self.index.ntotal - 1  # can never be <0
        self.faiss_to_image_ids[faiss_id] = vid

        if vid in self.id_map:
            self.id_map[vid].append(
                {
                    "label": label,
                    "camid": camid,
                    "faiss_id": faiss_id,
                    "path": path,
                }
            )
        else:
            self.id_map[vid] = [
                {
                    "label": label,
                    "camid": camid,
                    "faiss_id": faiss_id,
                    "path": path,
                }
            ]

        return vid

        # name = f"{vid}_{label}_{camid}"
        # path = os.path.join(self.root_dir, name)
        # prev = torchvision.io.read_image(path) if os.path.exists(path) else None

        # torchvision.io.write_png(image, os.path.join(self.root_dir, name))
        # return prev

    def __len__(self):
        return len(self.faiss_to_image_ids)  # we could also use `self.index.ntotal`

    def __getitem__(self, vid: Union[int, Tensor]):
        if isinstance(vid, Tensor):
            vid = vid.tolist()

        if vid not in self.id_map:
            raise KeyError(f"no vehicle found with id {vid}")

        return self.id_map[vid]

    @staticmethod
    def transform_embeddings(embs):
        embs = (
            F.normalize(torch.cat(embs, dim=0), p=2, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return embs.max(axis=0, keepdims=True)  # max pooling

    def search(self, embs, k=1, transform=True) -> Optional[int]:
        if k > 1:
            raise NotImplementedError  # todo: implement this

        embs_np = self.transform_embeddings(embs) if transform else embs
        distances, indices = self.index.search(embs_np, k=k)
        return (
            self.faiss_to_image_ids[indices[0][0]]
            if self.faiss_to_image_ids and distances[0][0] < THRESHOLD
            else None
        )

    def get_faiss_ids(self, vid: int) -> List[int]:
        return [m["faiss_id"] for m in self.id_map.get(vid, [])]

    def write_index(self):
        os.makedirs("faissdb", exist_ok=True)
        faiss.write_index(self.index, "faissdb/large.index")
        with open("faissdb/id_map.json", "w") as f:
            json.dump(self.id_map, f)
        with open("faissdb/faiss_to_image_ids.json", "w") as f:
            json.dump(self.faiss_to_image_ids, f)

    def read_index(self):
        self.index = faiss.read_index("faissdb/large.index")
        # json.dump converts int keys to strings, so we convert them back
        with open("faissdb/id_map.json", "r") as f:
            self.id_map = {int(k): v for k, v in json.load(f).items()}
        with open("faissdb/faiss_to_image_ids.json", "r") as f:
            self.faiss_to_image_ids = {int(k): v for k, v in json.load(f).items()}


def get_transform(data):
    return transforms.Compose(
        [
            transforms.Resize((data["y_length"], data["x_length"]), antialias=True),
            transforms.Normalize(data["n_mean"], data["n_std"]),
        ]
    )


def get_accuracy_metrics(
    data_q: CustomDataSet4Veri776_withviewpont, model, db: VehicleDB, scaler, data_g
):
    total_correct = 0
    total_expected = 0
    for i in tqdm(
        range(len(data_q)),
        desc="Query search (%)",
        bar_format="{l_bar}{bar:20}{r_bar}",
    ):
        (img, lbl, q_cam_id, q_view_id) = data_q[i]
        image = img.unsqueeze(0).to(device)
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, _, embs, _ = model(image, q_cam_id, q_view_id)
        else:
            _, _, embs, _ = model(image, q_cam_id, q_view_id)

        embs = db.transform_embeddings(embs)
        vid = db.search(embs[0].reshape(1, -1), transform=False)

        padded_lbl = f"{lbl:04}"

        actual = len([1 for m in db.id_map[vid] if m["label"] == lbl])
        expected = data_g.dataset.get_label_count(padded_lbl)

        total_correct += actual
        total_expected += expected

    return total_correct / total_expected


if __name__ == "__main__":
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
    parser.add_argument(
        "--update_index",
        action="store_true",
        help="Updates the faiss index",
    )
    args = parser.parse_args()

    with open(args.path_weights + "config.yaml", "r") as stream:
        data = yaml.safe_load(stream)

    model = get_model(data, torch.device("cpu"))

    # One of the saved weights last.pt best_CMC.pt best_mAP.pt
    path_weights = args.path_weights + "best_mAP.pt"

    transform = get_transform(data)

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
            transform=transform,
        )
        data_g = CustomDataSet4Veri776_withviewpont(
            data["gallery_list_file"],
            data["teste_dir"],
            data["train_keypoint"],
            data["test_keypoint"],
            is_train=False,
            transform=transform,
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
    path_weights = args.path_weights + "best_mAP.pt"

    try:
        model.load_state_dict(torch.load(path_weights, map_location="cpu"))
    except RuntimeError:
        tmp = torch.load(path_weights, map_location="cpu")
        tmp = OrderedDict((k.replace("module.", ""), v) for k, v in tmp.items())
        model.load_state_dict(tmp)

    model = model.to(device)
    model.eval()

    db = VehicleDB(not args.update_index)

    gf = []
    if args.update_index:
        # add all embeddings to the database
        for batch_idx, (image, label, camid, view_id) in enumerate(
            tqdm(
                data_g,
                desc="Gallery infer (%)",
                bar_format="{l_bar}{bar:20}{r_bar}",
            )
        ):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, batch_embs, _ = model(image, camid, view_id)
            else:
                _, _, batch_embs, _ = model(image, camid, view_id)

            del image

            # end_vec = []
            # for item in batch_embs:
            #     end_vec.append(F.normalize(item))
            # gf.append(torch.cat(end_vec, 1))

            batch_size = batch_embs[0].size(0)
            for i in range(batch_size):
                embs = [emb[i : i + 1, :] for emb in batch_embs]
                db.add(
                    label[i].item(),
                    camid[i].item(),
                    data_g.dataset.get_image_path(batch_size * batch_idx + i),
                    embs,
                    vid=None,
                )

        print("added all embeddings to index")
        db.write_index()

    # uncomment this to test for a single image
    # (query_image, q_id, q_cam_id, q_view_id) = data_q[int(args.query_image_id)]
    # image = query_image.unsqueeze(0).to(device)
    # if scaler:
    #     with torch.autocast(device_type="cuda", dtype=torch.float16):
    #         _, _, embs, _ = model(image, q_cam_id, q_view_id)
    # else:
    #     _, _, embs, _ = model(image, q_cam_id, q_view_id)

    # embs = db.transform_embeddings(embs)
    # vid = db.search(embs[0].reshape(1, -1), transform=False)
    # print(f"query image: {data_q.get_image_path(int(args.query_image_id))}")
    # print(f"vehicle id: {vid}, total matches: {len(db[vid])}\nmatches: {db[vid]}")

    print(get_accuracy_metrics(data_q, model, db, scaler, data_g))
