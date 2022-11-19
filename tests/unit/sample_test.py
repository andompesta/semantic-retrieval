from pyspark.sql import SparkSession
from pathlib import Path
import mlflow
from semantic_retrieval.tasks.sample_etl_task import SampleETLTask
from semantic_retrieval.tasks.sample_ml_task import SampleMLTask


def test_jobs(spark: SparkSession):
    print("Testing the ETL job")
    etl_job = SampleETLTask(spark)
    etl_job.launch()
    _count = spark.read.format("delta").load(
        str(
            Path(etl_job.args.base_path).joinpath(
                etl_job.args.database,
                etl_job.args.table
            ).absolute()
        )
    ).count()
    assert _count > 0
    print("Testing the ETL job - done")

    print("Testing the ML job")
    ml_job = SampleMLTask(spark)
    ml_job.launch()
    experiment = mlflow.get_experiment_by_name(
        str(
            Path(ml_job.args.base_path).joinpath(
                ml_job.args.experiment
            )
        )
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False
    print("Testing the ML job - done")
