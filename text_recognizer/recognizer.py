import argparse
from pathlib import Path
from typing import Sequence, Union
import numpy as np
from PIL import Image
import torch

from text_recognizer import util
from text_recognizer.stems.image import MNISTStem
ANS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿"]

STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "scripted_model"

MODEL_FILE = "model.pt"


class TextRecognizer:

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)
        self.stem = MNISTStem()

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """Predict/infer text in input image (which can be a file path, url, PIL.Image.Image or np.ndarray)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_pil = util.read_image_pil(image, grayscale=True)

        image_tensor = self.stem(image_pil).unsqueeze(axis=0)
        y_pred = self.model(image_tensor)[0]
        _, max_idx = torch.max(y_pred, 0)
        ans = ANS[max_idx]
        return ans




def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "filename",
        type=str,
        help="Name for an image file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage, an HDFS path, or any other resource locator supported by the smart_open library.",
    )
    args = parser.parse_args()

    text_recognizer = TextRecognizer()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()