from typing import Iterator, Tuple

from datetime import datetime
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from pyspark.sql import functions as F
from pyspark.sql.types import *

from src.common import Task
from semantic_retrieval.utils import ImageProcessor, encode


def get_resize_image_udf(
        img_size: Tuple[int, int]
):
    img_processor = ImageProcessor(
        img_size=img_size
    )

    def resize_image(
            dataframe_batch_iterator: Iterator[pd.DataFrame]
    ) -> Iterator[pd.DataFrame]:

        for dataframe_batch in dataframe_batch_iterator:
            paths = []
            contents = []
            shards = []
            imgs_array = []
            imgs_width = []
            imgs_height = []

            for row in dataframe_batch.itertuples():
                try:
                    img_array = img_processor(row.content)
                    imgs_array.append(encode(img_array))
                    imgs_width.append(img_size[0])
                    imgs_height.append(img_size[1])

                    paths.append(row.path)
                    contents.append(row.content)
                    shards.append(row.shard)
                except:
                    print(row)

            yield pd.DataFrame({
                "path": paths,
                "content": contents,
                "shard": shards,
                "img_array": imgs_array,
                "img_width": imgs_width,
                "img_height": imgs_height,
            })

    return resize_image


class FarFetchImageProcessing(Task):

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

        image_folder = str(
            base_path.joinpath(
                "images_cl",
                "images",
            )
        )

        content_output_path = str(
            base_path.joinpath(
                "images_cl",
                "content_",
            )
        )

        image_output_path = str(
            base_path.joinpath(
                "images_ready_",
            )
        )

        image_checkpoint_path = str(
            base_path.joinpath(
                "temp",
                "{}".format(int(datetime.now().timestamp()) // 1000)
            )
        )

        # create resize function
        resize_image_fn = get_resize_image_udf(img_size=(224, 224))

        # read images as stream
        images_stream = self.spark.readStream.format("cloudFiles").option(
            "cloudFiles.format", "binaryFile"
        ).option(
            "recursiveFileLookup", "true"
        ).option(
            "pathGlobFilter", "*.jpg"
        ).load(
            image_folder
        )

        output_stream = images_stream.select(
            F.col("path"),
            F.col("content")
        ).withColumn(
            "shard",
            F.regexp_replace(F.col("path"), "dbfs:\/Users\/scavallari\/farfetch\/images_cl\/images\/", "")
        ).withColumn(
            "shard",
            F.split(F.col("shard"), "\/").getItem(0)
        ).withColumn(
            "shard",
            F.col("shard").cast("int")
        )

        schema = StructType(output_stream.select("*").schema.fields + [
            StructField("img_array", BinaryType(), True),
            StructField("img_width", IntegerType()),
            StructField("img_height", IntegerType()),
        ])

        # apply resize function
        output_stream = output_stream.mapInPandas(resize_image_fn, schema)

        # Image data is already compressed, so you can turn off Parquet compression.
        self.spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        # write output stream
        autoload = output_stream.writeStream.format(
            "delta"
        ).option(
            "checkpointLocation",
            image_checkpoint_path
        ).partitionBy(
            "shard"
        ).trigger(
            once=True
        ).start(content_output_path)
        autoload.awaitTermination()


        # read parquet to get product_id
        farfetch = self.spark.read.format("parquet").load(
            str(
                base_path.joinpath(
                    "dataset",
                    "products.parquet"
                )
            )
        ).select(
            "product_id",
            "product_image_path",
        ).withColumn(
            "path",
            F.concat(
                F.lit(image_folder + "/"),
                F.col("product_image_path"),
            )
        )

        # join dataset and images on path
        farfetch.join(
            self.spark.read.format("delta").load(
                content_output_path
            ).select(
                "path",
                "img_array"
            ),
            on=["path"],
            how="inner"
        ).select(
            # select need fileds
            "product_id",
            "img_array",
        ).repartition(
            # repartition to reduce number of files
            512
        ).write.format("delta").mode(
            "overwrite"
        ).save(
            # save output
            image_output_path
        )


if __name__ == '__main__':
    task = FarFetchImageProcessing()
    task.launch()
