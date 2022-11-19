from src.common import Task
from sklearn.datasets import fetch_california_housing
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path


class SampleETLTask(Task):

    def parse_args(self):
        parser = ArgumentParser()

        parser.add_argument(
            "--table",
            type=str,
            default="sklearn_housing",
        )

        parser.add_argument(
            "--database",
            type=str,
            default="default",
        )

        parser.add_argument(
            "--base_path",
            type=str,
            default="data",
        )

        args, _ = parser.parse_known_args()

        print("\n\n-----------")
        for k, v in vars(args).items():
            print("{} \t {}".format(k, v))
        print("-----------\n\n")

        return args

    def _write_data(self):
        db = self.args.database
        table = self.args.table
        print(f"Writing housing dataset to {db}.{table}")
        _data: pd.DataFrame = fetch_california_housing(as_frame=True).frame
        df = self.spark.createDataFrame(_data)
        df.write.format(
            "delta"
        ).mode(
            "overwrite"
        ).save(
            str(
                Path(self.args.base_path).joinpath(
                    db,
                    table
                ).absolute()
            )
        )
        print("Dataset successfully written")

    def launch(self):
        print("Launching sample ETL task")
        self._write_data()
        print("Sample ETL task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = SampleETLTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
