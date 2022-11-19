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
        self.s3 = boto3.resource('s3', region_name='eu-central-1', )

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            spark = SparkSession.builder.getOrCreate()

        spark.conf.set(
            "spark.hadoop.fs.s3.impl",
            "com.databricks.s3a.S3AFileSystem",
        )
        spark.conf.set(
            "spark.hadoop.fs.s3a.acl.default",
            "BucketOwnerFullControl",
        )
        spark.conf.set(
            "spark.hadoop.fs.s3.impl",
            "com.databricks.s3a.S3AFileSystem",
        )
        spark.conf.set(
            "spark.hadoop.fs.s3a.canned.acl",
            "BucketOwnerFullControl",
        )
        spark.conf.set(
            "spark.hadoop.fs.s3.impl",
            "com.databricks.s3a.S3AFileSystem",
        )
        spark.conf.set(
            'spark.sql.session.timeZone',
            'Europe/Berlin',
        )
        return spark

    def get_dbutils(self):
        utils = get_dbutils(self.spark)

        if not utils:
            print("No DBUtils defined in the runtime")
        else:
            print("DBUtils class initialized")

        return utils

    def upload_to_s3(
            self,
            bucket: str,
            output_file_path: str,
            body: bytes
    ):
        s3object = self.s3.Object(
            bucket,
            output_file_path,
        )
        s3object.put(
            Body=body,
            ACL='bucket-owner-full-control',
        )

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
