import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import json

import torchvision
from PIL import ImageDraw

import torch
from omegaconf import OmegaConf
from facetorch import FaceAnalyzer


def load_config() -> OmegaConf:
    """Load the fully merged default configuration.

    This script is executed outside of Hydra's runtime which normally resolves
    the ``defaults`` list contained in ``conf/config.yaml``.  Using
    :func:`~omegaconf.OmegaConf.load` on that file would therefore result in a
    configuration object that is missing the ``analyzer`` section.  To mimic the
    behaviour of Hydra without pulling in the full dependency we directly load
    the pre-generated configuration found under ``conf/merged``.
    """

    config_file = (
        Path(__file__).resolve().parents[1] / "conf" / "merged" / "merged.config.yaml"
    )
    return OmegaConf.load(config_file)


def detect_faces(
    image_path: str,
    output_path: str,
    verbose: bool = False,
    compare: bool = False,
    montage_path: Optional[str] = None,
    json_path: Optional[str] = None,
) -> None:
"""Run face detection on a single image or all images in a folder.

    Parameters
    ----------
    image_path: str
        Path to the image that should be processed. If a directory is provided
        all images in that directory will be processed.
    output_path: str
        Where the resulting image with drawn detections is saved. If
        ``image_path`` is a directory this should also point to a directory
        where per-image outputs are written.
    montage_path: Optional[str]
        If provided, saves a thumbnail montage of all detected faces to this path.
    json_path: Optional[str]
        Path to save the detected face information as JSON.
    """
    cfg = load_config()
    analyzer = FaceAnalyzer(cfg.analyzer)

    img_path = Path(image_path)
    out_path = Path(output_path)
    responses = []

    img_files = [img_path]
    if img_path.is_dir():
        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            img_files.extend(sorted(img_path.glob(ext)))

    for img_file in img_files:
        if img_path.is_dir():
            if out_path.is_dir() or out_path.suffix == "":
                output_file = out_path / f"{img_file.stem}_detected{img_file.suffix}"
            else:
                output_file = out_path.parent / f"{img_file.stem}_detected{img_file.suffix}"
        else:
            output_file = out_path

        response = analyzer.run(
            path_image=str(img_file),
            batch_size=cfg.batch_size,
            fix_img_size=cfg.fix_img_size,
            return_img_data=cfg.return_img_data,
            include_tensors=cfg.include_tensors,
            path_output=str(output_file),
        )

        if response.img.nelement() > 0:
            img = torchvision.transforms.functional.to_pil_image(response.img.cpu())
            output_file.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_file)

        responses.append(response)

    if verbose:
        for resp in responses:
            print(resp)

    embeds: List[torch.Tensor] = []
    all_faces = []
    for resp in responses:
        for f in resp.faces:
            all_faces.append(f)
            if "embed" in f.preds:
                embeds.append(f.preds["embed"].logits)

    if compare and len(embeds) > 1:
        embed_tensor = torch.stack(embeds)
        dists = torch.cdist(embed_tensor, embed_tensor)
        print("Pairwise embedding distances:")
        for row in dists:
            print(" ".join(f"{val:.4f}" for val in row.tolist()))
    elif compare:
        print("No embeddings available for comparison")

    clusters = []
    if len(embeds) > 1:
        embed_tensor = torch.stack(embeds)
        dists = torch.cdist(embed_tensor, embed_tensor)
        threshold = 0.6
        adjacency = (dists < threshold).cpu().numpy()
        visited = set()
        for i in range(len(embeds)):
            if i in visited:
                continue
            stack = [i]
            cluster = []
            while stack:
                j = stack.pop()
                if j in visited:
                    continue
                visited.add(j)
                cluster.append(j)
                neighbors = [
                    k
                    for k in range(len(embeds))
                    if adjacency[j, k] and k not in visited
                ]
                stack.extend(neighbors)
            clusters.append(cluster)
    elif len(embeds) == 1:
        clusters.append([0])

    num_persons = len(clusters)
    print(f"Persons detected: {num_persons}")

    # compute distance from cluster center for each face and print cluster similarity tables
    face_distance = {}
    for c_idx, cluster in enumerate(clusters):
        c_embeds = torch.stack([embeds[i] for i in cluster])
        centroid = c_embeds.mean(dim=0)
        dists = torch.norm(c_embeds - centroid, dim=1)
        for j, face_idx in enumerate(cluster):
            face_distance[face_idx] = dists[j].item()
        sim_matrix = torch.cdist(c_embeds, c_embeds)
        print(f"Cluster {c_idx} similarity matrix:")
        for row in sim_matrix:
            print(" ".join(f"{val:.4f}" for val in row.tolist()))

    if json_path:

        def serialize(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            else:
                return obj

        faces_data = [serialize(asdict(face)) for face in all_faces]
        out = {"faces": faces_data, "clusters": clusters}
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

    if montage_path and len(all_faces) > 0:
        face_tensors = []
        for idx, face in enumerate(all_faces):
            if face.tensor.nelement() == 0:
                continue
            tensor = face.tensor.cpu()
            if idx in face_distance:
                pil_img = torchvision.transforms.functional.to_pil_image(tensor)
                draw = ImageDraw.Draw(pil_img)
                draw.text((2, 2), f"{face_distance[idx]:.2f}", fill=(255, 0, 0))
                tensor = torchvision.transforms.functional.to_tensor(pil_img)
            face_tensors.append(tensor)
        if face_tensors:
            grid = torchvision.utils.make_grid(
                face_tensors, nrow=min(8, len(face_tensors))
            )
            img = torchvision.transforms.functional.to_pil_image(grid)
            Path(montage_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(montage_path)

        montage_file = Path(montage_path)
        for idx, cluster in enumerate(clusters):
            cluster_tensors = []
            for i in cluster:
                if all_faces[i].tensor.nelement() == 0:
                    continue
                tensor = all_faces[i].tensor.cpu()
                if i in face_distance:
                    pil_img = torchvision.transforms.functional.to_pil_image(tensor)
                    draw = ImageDraw.Draw(pil_img)
                    draw.text((2, 2), f"{face_distance[i]:.2f}", fill=(255, 0, 0))
                    tensor = torchvision.transforms.functional.to_tensor(pil_img)
                cluster_tensors.append(tensor)
            if not cluster_tensors:
                continue
            grid = torchvision.utils.make_grid(
                cluster_tensors, nrow=min(8, len(cluster_tensors))
            )
            img = torchvision.transforms.functional.to_pil_image(grid)
            cluster_path = montage_file.with_name(
                f"{montage_file.stem}_cluster_{idx}{montage_file.suffix}"
            )
            cluster_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(cluster_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect faces in an image or folder")
    parser.add_argument(
        "--image",
        "-i",
        default="data/input/test.jpg",
        help="Path to an image file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/output/detected.png",
        help="Path for saving the output image or directory for multiple inputs",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print analyzer response",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare embeddings of detected faces",
    )
    parser.add_argument(
        "--montage",
        "-m",
        default="data/output/montage.png",
        help="Path for saving the montage of detected faces. Individual"
             " person montages will be saved with an appended '_cluster_X'"
             " before the file suffix.",
    )
    parser.add_argument(
        "--json",
        "-j",
        default="data/output/faces.json",
        help="Path for saving detected faces information as JSON",
    )
    args = parser.parse_args()
    detect_faces(
        args.image,
        args.output,
        verbose=args.verbose,
        compare=args.compare,
        montage_path=args.montage,
        json_path=args.json,
    )
