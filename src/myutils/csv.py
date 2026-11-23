import csv
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fp:
        yield from csv.reader(fp)


@contextmanager
def open_csv_writer(path: Path, fieldnames: Iterable[str] | None = None):
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)

        if fieldnames is not None and fp.tell() == 0:
            writer.writerow(fieldnames)

        yield writer
