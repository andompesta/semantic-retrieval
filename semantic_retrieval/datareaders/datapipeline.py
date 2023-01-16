from torchdata.datapipes.iter import (
    IterDataPipe,
    FSSpecFileLister,
)
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2 import (
    DataLoader2,
    PrototypeMultiProcessingReadingService,
)
import pyarrow.parquet as pq
from numpy import load, stack
from io import BytesIO
from typing import Union
from random import shuffle

@functional_datapipe("parquet_reader")
class ParquetReaderIter(IterDataPipe):

    def __init__(
        self,
        source_datapipe: IterDataPipe,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for files_chucnk in self.source_datapipe:
            # from parquet file to pyarrow Table
            chuck_table = pq.read_table(files_chucnk, memory_map=True)
            yield chuck_table


@functional_datapipe("arrow_batch")
class ArrowBatchIter(IterDataPipe):

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for table in self.source_datapipe:
            # from pyarrow Table to list of batches
            batches = table.to_batches(self.batch_size)

            if self.shuffle:
                shuffle(batches)

            yield from batches


@functional_datapipe("batch_decoder")
class BatchDecoderIter(IterDataPipe):

    def __init__(
        self,
        source_datapipe: IterDataPipe,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe

    @staticmethod
    def decode(value):
        # decode values accordinbg to NdarrayCodec
        memfile = BytesIO(value)
        return load(memfile)

    def __iter__(self):
        for batch in self.source_datapipe:
            # decode product id
            product_id = batch.column("product_id").to_numpy()
            # decode description
            description_ids = stack(
                [
                    self.decode(e.as_py())
                    for e in batch.column("description_ids")
                ],
                axis=0,
            )
            img_arrays = stack(
                [self.decode(e.as_py()) for e in batch.column("img_array")],
                axis=0,
            )

            yield (
                product_id,
                description_ids,
                img_arrays,
            )


def get_data_pipeline(
    dataset_location: str,
    batch_size: int,
    shuffle: bool,
    chunk_prefetch_buffer: int = 10,
    dataset_file_masks: Union[str, list(str)] = "*.parquet",
) -> IterDataPipe:
    pipeline = (
        FSSpecFileLister(
            dataset_location,
            dataset_file_masks,
        )
        # split parquet files across workers into shards
        .sharding_filter()
        # read chunk of file into a arrow Table
        .parquet_reader()
        # apply pre-fetching to read a new chunk while training
        .prefetch(chunk_prefetch_buffer)
        # apply batching
        .arrow_batch(batch_size, shuffle)
        # apply collate function
        .batch_decoder()
    )

    return pipeline


def get_multi_processing_dataloader(
    dataset_location: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    **kwargs,
) -> DataLoader2:
    reading_serivce = PrototypeMultiProcessingReadingService(
        num_workers=num_workers,
        multiprocessing_context="fork",
    )

    pipeline = get_data_pipeline(
        dataset_location=dataset_location,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )

    dataloader = DataLoader2(
        pipeline,
        reading_service=reading_serivce,
    )

    return dataloader
