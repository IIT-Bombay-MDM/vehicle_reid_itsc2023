import argparse
from typing import OrderedDict

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO

from db import VehicleDB, get_transform
from models.models import MBR_model
from processor import get_model


def train_extraction_model():
    model = YOLO("yolov9m.pt")
    model.train(data="data.yaml", epochs=50, plots=True)


def load_extraction_model(weights):
    return YOLO(weights)


def extract_vehicles(model, data):
    return model(data)


def get_embeddings(img: Image, model: MBR_model, transform, scaler, device):
    img = pil_to_tensor(img)
    img = transform(img)

    img = img.to(device)

    if scaler:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, _, embs, _ = model(img)
    else:
        _, _, embs, _ = model(img)

    return embs


def add_extracts(
    result, db: VehicleDB, label_start: int, model: MBR_model, transform, scaler, device
):
    for detection in result:
        bgr_array = detection.plot()
        rgb_array = bgr_array[..., ::-1]
        output_image = Image.fromarray(rgb_array)

        embs = get_embeddings(output_image, model, transform)

        db.add(label_start, 0, None, embs)
        label_start += 1


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

    extraction_model = load_extraction_model("extraction.pt")

    # todo: get data
    data = []

    results = extract_vehicles(extraction_model, data)

    add_extracts(results, db, 0, model, transform, scaler, device)

    # `db` can now be used to query
