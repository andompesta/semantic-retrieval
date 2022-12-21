import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pyspark.sql import functions as F
from pyspark.sql.types import *

from semantic_retrieval.common import Task

class FarFetchModelTraining(Task):

    def parse_args(self):
        parser = ArgumentParser()

        parser.add_argument(
            "--dataset_base_path",
            type=lambda x: Path(x),
            required=True,
            help="basepath of the raw dataset",
        )

        args, _ = parser.parse_known_args()

        print("\n\n-----------")
        for k, v in vars(args).items():
            print("{} \t {}".format(k, v))
        print("-----------\n\n")

        return args

    def launch(self):
        
        base_path = Path(self.args.dataset_base_path)
        train_dataset_path = base_path.joinpath(
            "training"
        )
        validation_dataset_path = base_path.joinpath(
            "validation"
        )

        model = CLIP(
            
        )

        with get_farfetch_dataloader(
            path="file:///dbfs" + train_dataset_path.as_posix()
        ) as train_dl:




if __name__ == '__main__':
    task = FarFetchDatasetPreparation()
    task.launch()
