import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import json

import torchvision

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
    """Run face detection on a single image.

    Parameters
    ----------
    image_path: str
        Path to the image that should be processed.
    output_path: str
        Where the resulting image with drawn detections is saved.
    montage_path: Optional[str]
        If provided, saves a thumbnail montage of all detected faces to this path.
    json_path: Optional[str]
        Path to save the detected face information as JSON.
    """
    cfg = load_config()
    analyzer = FaceAnalyzer(cfg.analyzer)
    response = analyzer.run(
        path_image=image_path,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=output_path,
    )
    if response.img.nelement() > 0:
        img = torchvision.transforms.functional.to_pil_image(response.img.cpu())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
    if verbose:
        print(response)

    embeds: List[torch.Tensor] = [
        face.preds["embed"].logits for face in response.faces if "embed" in face.preds
    ]

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
        print("Clusters (threshold=0.6):")
        for idx, c in enumerate(clusters):
            print(f"  Cluster {idx}: {c}")

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

        faces_data = [serialize(asdict(face)) for face in response.faces]
        out = {"faces": faces_data, "clusters": clusters}
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

    if montage_path and len(response.faces) > 0:
        face_tensors = [
            face.tensor.cpu() for face in response.faces if face.tensor.nelement() > 0
        ]
        if face_tensors:
            grid = torchvision.utils.make_grid(
                face_tensors, nrow=min(8, len(face_tensors))
            )
            img = torchvision.transforms.functional.to_pil_image(grid)
            Path(montage_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(montage_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect faces in a single image")
    parser.add_argument(
        "--image",
        "-i",
        default="data/input/test.jpg",
        help="Path to an image file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/output/detected.png",
        help="Path for saving the output image",
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
        help="Path for saving the montage of detected faces",
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
