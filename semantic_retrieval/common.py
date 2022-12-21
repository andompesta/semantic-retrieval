from abc import ABC, abstractmethod
from pyspark.sql import SparkSession
import boto3


def get_dbutils(
    spark: SparkSession,
):  # please note that this function is used in mocking by its name
    try:
        from pyspark.dbutils import DBUtils  # noqa

        if "dbutils" not in locals():
            utils = DBUtils(spark)
            return utils
        else:
            return locals().get("dbutils")
    except ImportError:
        return None


class Task(ABC):

    def __init__(self, spark=None):
        self.args = self.parse_args()
        self.spark = self._prepare_spark(spark)
        self.dbutils = self.get_dbutils()
        self.s3 = boto3.resource(
            's3',
            region_name='eu-central-1',
        )

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            spark = SparkSession.builder.getOrCreate()
        return spark

    def get_dbutils(self):
        utils = get_dbutils(self.spark)

        if not utils:
            print("No DBUtils defined in the runtime")
        else:
            print("DBUtils class initialized")

        return utils

    @abstractmethod
    def parse_args(self):
        pass

    @abstractmethod
    def launch(self):
        """
        Main method of the job.
        :return:
        """
        pass
