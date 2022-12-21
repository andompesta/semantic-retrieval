import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
from pyspark.sql import functions as F
from pyspark.sql.types import *

from semantic_retrieval.common import Task
from semantic_retrieval.datareaders import get_farfetch_dataloader
from semantic_retrieval.utils import compute_warmup_steps
from semantic_retrieval.model import CLIP
from semantic_retrieval.model.config import LargeClipConfig

class FarFetchModelTraining(Task):

    def parse_args(self):
        parser = ArgumentParser()
        parser.add_argument(
            "--run_name",
            required=True,
            help="run name used for logging",
        )

        parser.add_argument(
            "--base_path",
            type=lambda x: Path(x),
            required=True,
            help="basepath of the raw dataset",
        )

        parser.add_argument(
            "--device",
            type=str,
            default="mps",
        )

        parser.add_argument(
            "--num_epocs",
            type=int,
            default=10,
        )
        
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
        )

        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=1000,
        )

        parser.add_argument(
            "--batches_per_epoch",
            type=int,
            default=10000,
        )

        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
        parser.add_argument("--freeze_layer", default=3, type=int)
        parser.add_argument("--optim_method", default="adam")
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--lr", default=5e-5, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)

        args = compute_warmup_steps(args)
        args, _ = parser.parse_known_args()

        print("\n\n-----------")
        for k, v in vars(args).items():
            print("{} \t {}".format(k, v))
        print("-----------\n\n")

        return args

    def launch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        base_path = self.args.base_path
        train_dataset_path = base_path.joinpath(
            "training"
        )
        validation_dataset_path = base_path.joinpath(
            "validation"
        )

        model_path = base_path.joinpath(
            "checkpoinst",
            "ViT-L-14@336px.pt",
        )

        state_dict = torch.load(
            "/dbfs" + model_path.as_posix(),
            map_location="cpu",
        )

        model = CLIP(
            **LargeClipConfig()
        ).load_state_dict(
            state_dict,
            strict=True,
        )



        with get_farfetch_dataloader(
            path="file:///dbfs" + train_dataset_path.as_posix()
        ) as train_dl:




if __name__ == '__main__':
    task = FarFetchDatasetPreparation()
    task.launch()
