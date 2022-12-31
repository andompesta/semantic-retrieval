from petastorm.reader import make_reader
from petastorm.pytorch import DataLoader
from contextlib import contextmanager
from typing import Optional

@contextmanager
def get_farfetch_dataloader(
    path: str,
    batch_size: int,
    reader_pool_type: str,
    workers_count: int = 5,
    num_epochs: Optional[int] = None,
    shuffle: bool = True,
):
    with make_reader(
            path,
            reader_pool_type=reader_pool_type,
            workers_count=workers_count,
            num_epochs=num_epochs,
            shuffle_row_groups=shuffle,
    ) as reader:
        dataloader = DataLoader(
            reader,
            batch_size=batch_size,
        )
        yield dataloader


@contextmanager
def get_single_batch_farfetch_dataloader(
    path: str,
    batch_size: int,
    reader_pool_type: str,
    workers_count: int = 5,
):
    with make_reader(
            path,
            reader_pool_type=reader_pool_type,
            workers_count=workers_count,
            num_epochs=1,
            # shuffle_rows=shuffle,
            shuffle_row_groups=False,
    ) as reader:
        dataloader = DataLoader(
            reader,
            batch_size=batch_size,
        )
        for batch in dataloader:
            break

    yield [batch]
