from contextlib import contextmanager
from io import BytesIO
from numpy import save, load
from pathlib import Path
from shutil import rmtree

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import tempfile


def encode(value):
    # encode values according to NdarrayCodec
    # https://petastorm.readthedocs.io/en/latest/_modules/petastorm/codecs.html?highlight=bytearray
    memfile = BytesIO()
    save(memfile, value)
    return bytearray(memfile.getvalue())


def decode(value):
    # decode values accordinbg to NdarrayCodec
    memfile = BytesIO(value)
    return load(memfile)


@contextmanager
def get_spark():
    warehouse_dir = Path(tempfile.TemporaryDirectory().name)
    try:
        _builder = SparkSession.builder.master(
            "local[1]"
        ).config(
            "spark.hive.metastore.warehouse.dir", warehouse_dir.as_uri()
        ).config(
            "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
        ).config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )

        spark: SparkSession = configure_spark_with_delta_pip(_builder).getOrCreate()
        print("Spark session configured")
        yield spark
    finally:
        print("Shutting down Spark session")
        spark.stop()
        if Path(warehouse_dir).exists():
            rmtree(warehouse_dir)
