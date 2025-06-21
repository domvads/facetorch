import argparse
from omegaconf import OmegaConf
from facetorch import FaceAnalyzer


def load_config():
    """Load the default merged configuration."""
    return OmegaConf.load("conf/merged/merged.config.yaml")


def detect_faces(image_path: str, output_path: str) -> None:
    """Run face detection on a single image.

    Parameters
    ----------
    image_path: str
        Path to the image that should be processed.
    output_path: str
        Where the resulting image with drawn detections is saved.
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
    print(response)


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
    args = parser.parse_args()
    detect_faces(args.image, args.output)
