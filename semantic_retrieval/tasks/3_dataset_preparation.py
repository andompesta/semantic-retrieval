import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pyspark.sql import functions as F
from pyspark.sql.types import *

from semantic_retrieval.common import Task
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.codecs import (
    ScalarCodec,
    NdarrayCodec,
)
from petastorm.unischema import (
    Unischema,
    UnischemaField,
)


class FarFetchDatasetPreparation(Task):

    @staticmethod
    def tuple_type(arg: str):
        arg = arg.replace("(", "").replace(")", "")
        mapped_int = map(int, arg.split(","))
        return tuple(mapped_int)

    def parse_args(self):
        parser = ArgumentParser()

        parser.add_argument(
            "--dataset_base_path",
            type=str,
            required=True,
            help="basepath of the raw dataset",
        )

        parser.add_argument(
            "--image_size",
            type=FarFetchDatasetPreparation.tuple_type,
            default="(3, 336, 336)",
        )

        parser.add_argument(
            "--description_size",
            type=FarFetchDatasetPreparation.tuple_type,
            default="(77)",
        )

        args, _ = parser.parse_known_args()

        print("\n\n-----------")
        for k, v in vars(args).items():
            print("{} \t {}".format(k, v))
        print("-----------\n\n")

        return args

    def launch(self):
        self.spark.conf.set(
            "spark.sql.parquet.compression.codec",
            "gzip",
        )

        base_path = Path(self.args.dataset_base_path)

        text_path = str(base_path.joinpath("description_ready_cpu"))

        image_path = str(base_path.joinpath("images_ready_large"))

        # get descriptions
        text = self.spark.read.format("delta").load(text_path)

        # get image preprocessed
        imgs = self.spark.read.format("delta").load(image_path).select(
            "product_id",
            "img_array",
        )

        # join information
        dataset = text.join(
            imgs,
            on=["product_id"],
            how="inner",
        )

        # The schema defines how the dataset schema looks like
        far_fetch_schema = Unischema('FarFetchSchema', [
            UnischemaField(
                'product_id',
                np.int64,
                (),
                ScalarCodec(IntegerType()),
                False,
            ),
            UnischemaField(
                'description_ids',
                np.int64,
                self.args.description_size,
                NdarrayCodec(),
                False,
            ),
            UnischemaField(
                'img_array',
                np.float64,
                self.args.image_size,
                NdarrayCodec(),
                False,
            ),
        ])

        # split train validation dataset
        dataset = dataset.withColumn(
            "dataset",
            F.when(
                (F.abs(F.hash(F.col('product_id'))) % 16) < F.lit(14),
                'training',
            ).otherwise(
                'validation'
            )
        )

        for partition_name in ["training", "validation"]:
            # get dataset split
            dataset_partition = dataset.filter(
                F.col("dataset") == partition_name
            ).drop(
                # dataset is not part of the schema, thus remove it
                "dataset"
            )

            if partition_name == "training":
                dataset_partition = dataset_partition.repartition(2000)
            else:
                dataset_partition = dataset_partition.repartition(200)

            # partition output path
            output_path = str(
                base_path.joinpath(
                    "datasets",
                    "completed_large",
                    partition_name,
                )
            ).replace("dbfs:", "file:///dbfs")

            # materialize dataset in petastomr format
            with materialize_dataset(
                    self.spark,
                    output_path,
                    far_fetch_schema,
                    500,
            ):
                dataset_partition.write.mode("overwrite").parquet(output_path)


if __name__ == '__main__':
    task = FarFetchDatasetPreparation()
    task.launch()
