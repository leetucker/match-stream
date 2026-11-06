"""Gzip the reference outputs the tests read: tests/data/ref_*.csv -> ref_*.csv.gz
(ref_data.csv, the input every script reads, stays plain).

Run it after regenerating outputs with the R and Stata scripts, which write plain
CSVs: ``python tests/reference/compress.py``. The archives carry no timestamp, so
unchanged content gives an identical file.
"""

import gzip
import shutil
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    for path in sorted(DATA.glob("ref_*.csv")):
        if path.name == "ref_data.csv":
            continue
        out = path.with_name(path.name + ".gz")
        with path.open("rb") as src, out.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                               compresslevel=9) as dst:
                shutil.copyfileobj(src, dst)
        path.unlink()
        print(f"{path.name} -> {out.name}")


if __name__ == "__main__":
    main()
