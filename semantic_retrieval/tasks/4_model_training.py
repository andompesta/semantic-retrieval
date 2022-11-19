import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pyspark.sql import functions as F
from pyspark.sql.types import *

from semantic_retrieval.common import Task
from petastorm import make_reader

class FarFetchModelTraining(Task):

    def parse_args(self):
        parser = ArgumentParser()

        parser.add_argument(
            "--dataset_base_path",
            type=str,
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





if __name__ == '__main__':
    task = FarFetchDatasetPreparation()
    task.launch()
