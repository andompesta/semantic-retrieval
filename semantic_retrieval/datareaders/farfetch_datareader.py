from pyspark.sql import DataFrame, functions as F, SparkSession
from pyspark.sql.types import *
from pathlib import Path


def get_farfetch(
        path: Path,
        spark: SparkSession,
) -> DataFrame:
    farfetch = spark.read.format("parquet").load(
        str(path)
    )

    # parse attributes
    farfetch = farfetch.withColumn(
        "product_attributes",
        F.from_json(
            F.col("product_attributes"),
            ArrayType(
                StructType([
                    StructField("attribute_name", StringType()),
                    StructField("attribute_values", ArrayType(StringType())),
                ])
            )
        )
    )
    return farfetch
