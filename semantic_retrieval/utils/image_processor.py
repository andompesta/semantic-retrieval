from typing import Tuple

import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.transforms import (Compose, Resize, CenterCrop, ToTensor,
                                    Normalize)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class ImageProcessor(object):

    def __init__(
        self,
        img_size: Tuple[int, int],
    ):
        super().__init__()
        self.img_size = img_size
        self.img_out_size = (3,) + img_size

        self.transformation = Compose([
            Resize(
                size=self.img_size,
                interpolation=BICUBIC,
            ),
            # CenterCrop(self.img_size),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __call__(self, data: bytes) -> np.array:
        with Image.open(BytesIO(data)) as img:
            img = self.transformation(img)
            img_array = img.detach().cpu().numpy()

        assert img_array.shape == self.img_out_size, "image not properly formatted"

        return img_array
