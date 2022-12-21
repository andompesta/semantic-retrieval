from petastorm.reader import make_reader
from petastorm.pytorch import DataLoader
from contextlib import contextmanager


@contextmanager
def get_farfetch_dataloader(
    path: str,
    batch_size: int,
    reader_pool_type: str,
    workers_count: int = 5,
    shuffle: bool = True,
):
    with make_reader(
            path,
            reader_pool_type=reader_pool_type,
            workers_count=workers_count,
            num_epochs=1,
            shuffle_rows=shuffle,
            shuffle_row_groups=shuffle,
    ) as reader:
        dataloader = DataLoader(
            reader,
            batch_size=batch_size,
        )
        yield dataloader
