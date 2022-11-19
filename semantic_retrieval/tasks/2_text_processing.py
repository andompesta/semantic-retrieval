from typing import (
    Sequence,
    Iterator,
    Callable,
)

import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pyspark.sql import functions as F, DataFrame
from pyspark.sql.types import *

from semantic_retrieval.utils import (
    SimpleTokenizer,
    encode,
)
from src.common import Task


def verbalize_description(pdf):
    def verbalize_from_list(
            verbs
    ):
        if len(verbs) == 1:
            return str(verbs[0]).lower()
        else:
            return (", ".join(verbs[:-1]) + " and {}".format(verbs[-1])).lower()

    def build_pattern(
            base_pattern,
            fillers
    ):
        if fillers is None or len(fillers) == 0:
            return ""
        if fillers[0] is None:
            return ""

        return (base_pattern + " " + verbalize_from_list(fillers)).strip()

    patterns = np.array([
        "a {category} made by {brand} {color_pattern} {attribute_pattern} {gender_pattern}. {short_description}",
        "a {category} made by {brand} {attribute_pattern}; {color_pattern} {gender_pattern}. {short_description}",
        "{color} {category} {attribute_pattern} from {brand} {gender_pattern}. {short_description}",
    ])

    random_idx = np.random.randint(
        0,
        patterns.shape[0],
        size=pdf.shape[0]
    )

    patterns = patterns[random_idx]

    compiled_patterns = []
    for row, pattern in zip(pdf.itertuples(), patterns):
        formatted = pattern.format(
            **dict(
                category=row.desc_category,
                brand=row.desc_brand,
                gender_pattern=build_pattern("for", [row.desc_gender]),
                color_pattern=build_pattern("color", [row.desc_color]),
                color=build_pattern("", [row.desc_color]),
                attribute_pattern=build_pattern("with", row.attribute_values),
                short_description=build_pattern("", [row.product_short_description])
            )
        ).strip()
        compiled_patterns.append(formatted)

    return pd.DataFrame({
        "product_id": pdf.product_id.values,
        "description": np.array(compiled_patterns)
    })


def get_preprocess_data_fn(
        vocab_path: str
) -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:
    tokenizer = SimpleTokenizer(
        vocab_path,
        dtype=int,
    )

    def preprocess_data(
            dataframe_batch_iterator: Iterator[pd.DataFrame]
    ) -> Iterator[pd.DataFrame]:

        for dataframe_batch in dataframe_batch_iterator:
            product_ids = []
            descriptions_ids = []
            for row in dataframe_batch.itertuples():
                product_ids.append(int(row.product_id))

                # tokenize description
                description_ids = tokenizer(row.description)
                # description_ids = description_ids.astype(np.int64)
                # enconde numpy array as byte-array
                descriptions_ids.append(encode(description_ids))

            yield pd.DataFrame({
                "product_id": product_ids,
                "description_ids": descriptions_ids,
            })

    return preprocess_data


class FarFetchTextProcessing(Task):

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

        self.spark.conf.set(
            "spark.sql.parquet.columnarReaderBatchSize",
            100
        )

        text_output_path = str(
            base_path.joinpath(
                "description_ready",
            )
        )

        farfetch = self.spark.read.format("parquet").load(
            str(
                base_path.joinpath(
                    "dataset",
                    "products.parquet",
                )
            )
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

        farfetch = self.cateogry_filtering(farfetch)
        farfetch = self.handle_none(farfetch)
        farfetch_attributes = self.get_attributes(farfetch)

        farfetch_verb = self.get_verbalizer(
            df=farfetch,
            attributes=farfetch_attributes
        ).withColumn(
            "shard",
            F.monotonically_increasing_id() % 150
        )

        # generate description of product
        farfetch_description = farfetch_verb.groupBy(
            "shard"
        ).applyInPandas(
            verbalize_description,
            schema=StructType([
                StructField("product_id", IntegerType()),
                StructField("description", StringType())
            ])
        ).drop("shard")

        preprocess_data_fn = get_preprocess_data_fn(
            str(
                base_path.joinpath(
                    "checkpoints",
                    "bpe_simple_vocab_16e6.txt.gz"
                )
            ).replace("dbfs:", "/dbfs")
        )

        # generate text encoding and image preprocessing
        farfetch_description = farfetch_description.mapInPandas(
            preprocess_data_fn,
            StructType([
                StructField("product_id", IntegerType()),
                StructField("description_ids", BinaryType())
            ])
        )

        farfetch_description.write.format("delta").save(
            text_output_path
        )

    def cateogry_filtering(
            self,
            df: DataFrame
    ) -> DataFrame:
        # get categories
        product_categories = df.groupby(
            "product_family",
            "product_category"
        ).agg(
            F.collect_set(F.col("product_sub_category")).alias("product_sub_categories"),
            F.count("*").alias("product_category_cout")
        )

        # filter categories
        df = df.join(
            product_categories.filter(F.col("product_category_cout") <= 10).select(
                "product_category"
            ),
            on="product_category",
            how="left_anti"
        ).filter(
            # remove secondary product-families
            (~F.col("product_family").isin([
                "Demi-Fine Jewellery",
                "Fine Jewellery",
                "Homeware",
                "Jewellery",
                "Watches"
            ]))
        )

        return df

    def get_attributes(
            self,
            df: DataFrame
    ) -> DataFrame:
        """extract product attribites from datasets

        :param df: farfetch dataset
        :type df: DataFrame
        :return: products attributes
        :rtype: DataFrame
        """

        # expand attributes
        attributes = df.select(
            "product_id",
            F.explode("product_attributes").alias("product_attributes")
        ).select(
            "product_id",
            "product_attributes.*"
        )

        attributes_stats = attributes.groupBy(
            "attribute_name"
        ).agg(
            F.collect_set("attribute_values").alias("attribute_values"),
            F.count("*").alias("count")
        ).withColumn(
            "attribute_values",
            F.array_distinct(F.flatten("attribute_values"))
        ).toPandas()

        # filter attributres that does not appears more that 50 times
        attributes = attributes.filter(
            ~F.col("attribute_name").isin(
                attributes_stats[attributes_stats["count"] < 50]["attribute_name"].values.tolist()
            )
        ).filter(
            # remove information about item conditions as should be all new
            F.col("attribute_name") != "Condition"
        ).filter(
            # remove information about item's occasion
            F.col("attribute_name") != "Occasion"
        ).filter(
            # remove attributes with multiple values as are only few and usualy confusing
            F.size(F.col("attribute_values")) <= 1
        ).withColumn(
            # flatten attribute_values
            "attribute_values",
            F.element_at("attribute_values", 1).alias("attribute_values")
        )

        return attributes

    def get_verbalizer(
            self,
            df: DataFrame,
            attributes: DataFrame,
    ) -> DataFrame:
        # prepare variables for verbalizer
        df = df.withColumn(
            "desc_brand",
            F.col("product_brand")
        ).withColumn(
            "desc_category",
            F.when(
                F.col("product_sub_category").isNotNull(), F.lower(F.col("product_sub_category"))
            ).otherwise(
                F.lower(F.col("product_category"))
            )
        ).withColumn(
            "desc_gender",
            F.when(
                F.col("product_gender").isin(["MEN", "WOMEN"]), F.lower(F.col("product_gender"))
            ).otherwise(
                None
            )
        ).withColumn(
            "desc_color",
            F.lower(F.col("product_main_colour"))
        )

        return df.select([
            "product_id",
            "product_family",
            "desc_category",
            "desc_brand",
            "desc_gender",
            "desc_color",
            "product_image_path",
            "product_short_description"
        ]).join(
            attributes.groupBy("product_id").agg(
                F.collect_set("attribute_values").alias("attribute_values")
            ),
            on="product_id",
            how="left"
        )

    def handle_none(
            self,
            df: DataFrame,
            cols: Sequence[str] = (
                    "product_brand",
                    "product_category",
                    "product_family",
                    "product_sub_category",
                    "product_gender",
                    "product_main_colour",
                    "product_short_description",
                    "product_image_path",
            ),
    ) -> DataFrame:
        for col in cols:
            df = df.withColumn(
                col,
                F.when(
                    (F.col(col) == "N/D") | (F.col(col) == "[N/D]") | (F.col(col) == "None"),
                    None
                ).otherwise(
                    F.col(col)
                )
            )
        return df


if __name__ == '__main__':
    task = FarFetchTextProcessing()
    task.launch()
