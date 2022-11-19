from argparse import ArgumentParser
from pathlib import Path

from src.common import Task
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import mlflow.sklearn
import mlflow


class SampleMLTask(Task):
    TARGET_COLUMN: str = "MedHouseVal"

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

        parser.add_argument(
            "--experiment",
            default="sample_experiment"
        )

        args, _ = parser.parse_known_args()

        print("\n\n-----------")
        for k, v in vars(args).items():
            print("{} \t {}".format(k, v))
        print("-----------\n\n")

        return args

    def _read_data(self) -> pd.DataFrame:
        db = self.args.database
        table = self.args.table
        print(f"Reading housing dataset from {db}.{table}")
        _data: pd.DataFrame = self.spark.read.format(
            "delta"
        ).load(
            str(
                Path(self.args.base_path).joinpath(
                    db,
                    table
                ).absolute()
            )
        ).toPandas()
        print(f"Loaded dataset, total size: {len(_data)}")
        return _data

    @staticmethod
    def get_pipeline() -> Pipeline:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ('random_forest', RandomForestRegressor())
        ])
        return pipeline

    def _train_model(self):
        mlflow.sklearn.autolog()
        pipeline = self.get_pipeline()
        data = self._read_data()
        X = data.drop(self.TARGET_COLUMN, axis=1)
        y = data[self.TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2_result = r2_score(y_test, y_pred)
        mlflow.log_metric("r2", r2_result)

    def launch(self):
        print("Launching sample ML task")
        mlflow.set_experiment(
            str(
                Path(self.args.base_path).joinpath(
                    self.args.experiment
                )
            )
        )
        self._train_model()
        print("Sample ML task finished")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = SampleMLTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
