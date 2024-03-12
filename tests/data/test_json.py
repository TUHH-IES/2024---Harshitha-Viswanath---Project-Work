import os
import tempfile
import unittest
from pathlib import Path

import polars as pl
from agenc.data import JsonDataLoader
from polars.testing import assert_frame_equal


class TestJsonDataLoader(unittest.TestCase):
    def test_single_sample(self) -> None:
        json_content = """
        {
            "A": 1,
            "B": 2,
            "C": 3
        }
        """

        data = pl.DataFrame({"A": 1, "B": 2, "C": 3})

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                f.write(bytes(json_content, "UTF-8"))
                f.close()

                dataloader = JsonDataLoader(path=Path(f.name))
                loaded_data = dataloader.load()
                assert_frame_equal(loaded_data, data)
            finally:
                f.close()
                os.remove(f.name)

    def test_multiple_samples(self) -> None:
        json_content = """
        [
            {
                "A": 1,
                "B": 2,
                "C": 3
            },
            {
                "A": 4,
                "B": 5,
                "C": 6
            }
        ]
        """

        data = pl.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]})

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                f.write(bytes(json_content, "UTF-8"))
                f.close()

                dataloader = JsonDataLoader(path=Path(f.name))
                loaded_data = dataloader.load()
                assert_frame_equal(loaded_data, data)
            finally:
                f.close()
                os.remove(f.name)


if __name__ == "__main__":
    unittest.main()