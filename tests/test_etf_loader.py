import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd

from utils import duckdb_manager, loader


class EtfLoaderTest(unittest.TestCase):
    def test_etf_table_and_adjustment(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "test.duckdb")
            cache_path = Path(directory) / "cum_factor.parquet"
            connect = lambda: duckdb.connect(db_path)
            with (
                mock.patch.object(duckdb_manager, "DB_PATH", db_path),
                mock.patch.object(loader, "DB_PATH", db_path),
                mock.patch.object(loader, "CACHE_PATH", cache_path),
                mock.patch.object(loader, "get_conn", connect),
            ):
                duckdb_manager.init_database()
                conn = connect()
                table_info = conn.execute(
                    "PRAGMA table_info('etf')"
                ).fetchall()
                columns = [row[1] for row in table_info]
                self.assertEqual(columns, duckdb_manager.HISTORY_COLS)
                self.assertEqual(
                    [row[1] for row in table_info if row[5]],
                    ["symbol", "date"],
                )
                rows = [
                    ("510300", "2026-01-05", 10, 12, 9, 11, 100, 1000),
                    ("510300", "2026-01-06", 11, 13, 10, 12, 200, 2000),
                    ("510500", "2026-01-05", 20, 22, 19, 21, 300, 3000),
                ]
                conn.executemany(
                    "INSERT INTO etf VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
                )
                conn.executemany(
                    "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [("600000", *row[1:]) for row in rows[:2]],
                )
                conn.executemany(
                    "INSERT INTO ex_factors VALUES (?, ?, ?)",
                    [
                        ("510300", "2026-01-05", 1.0),
                        ("510300", "2026-01-06", 1.1),
                        ("600000", "2026-01-05", 1.0),
                        ("600000", "2026-01-06", 1.1),
                    ],
                )
                conn.close()

                raw = loader.load_etf_prices(
                    "2026-01-05",
                    "2026-01-06",
                    adjust="none",
                    symbols="510300.SH",
                )
                adjusted = loader.load_etf_prices(
                    "2026-01-05",
                    "2026-01-06",
                    adjust="backward",
                    symbols=["510300"],
                )
                stock = loader.load_prices(
                    "2026-01-05",
                    "2026-01-06",
                    adjust="backward",
                    force_refresh_cache=True,
                )

                self.assertEqual(set(raw.index.get_level_values("symbol")), {"510300"})
                pd.testing.assert_frame_equal(
                    adjusted.reset_index(drop=True),
                    stock.reset_index(drop=True),
                )
                pd.testing.assert_frame_equal(
                    adjusted[["volume", "amount"]],
                    raw[["volume", "amount"]],
                )
                missing_factor = loader.load_etf_prices(
                    "2026-01-05",
                    "2026-01-06",
                    adjust="backward",
                    symbols=["510500.SH"],
                )
                self.assertEqual(missing_factor.iloc[0]["close"], 21)


if __name__ == "__main__":
    unittest.main()
