import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import duckdb

import update_database


OBSERVATION_HEADER = [
    "id",
    "decimalLatitude",
    "decimalLongitude",
    "taxonID",
    "taxonRank",
    "eventDate",
]


class DatabaseUpdaterTests(unittest.TestCase):
    def write_zip(self, path: Path, member: str, rows: list[list[str]]) -> None:
        text = "\n".join(",".join(row) for row in rows) + "\n"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member, text)

    def test_update_builds_both_databases_and_cleans_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name)
            observations_zip = directory / update_database.OBSERVATIONS_ARCHIVE_NAME
            taxonomy_zip = directory / update_database.TAXONOMY_ARCHIVE_NAME
            self.write_zip(
                observations_zip,
                "nested/observations.csv",
                [
                    OBSERVATION_HEADER,
                    ["1", "37.12345", "-122.98765", "10", "species", "2025-02-03T12:00:00Z"],
                    ["2", "999", "-122", "10", "species", "2025-02-04"],
                ],
            )
            self.write_zip(
                taxonomy_zip,
                "taxa.csv",
                [
                    ["taxonID", "parentNameUsageID", "scientificName"],
                    ["https://www.inaturalist.org/taxa/10", "", "Root"],
                    [
                        "https://www.inaturalist.org/taxa/11",
                        "https://www.inaturalist.org/taxa/10",
                        "Child",
                    ],
                ],
            )
            cache_path = directory / "taxon_cache.json"
            cache_path.write_text(
                json.dumps({"Root": 10, "Root_descendants": [10, 11]}),
                encoding="utf-8",
            )

            observation_stats, taxonomy_stats = update_database.update_databases(
                observations_zip, taxonomy_zip, directory
            )

            self.assertEqual(observation_stats["rows"], 1)
            self.assertEqual(taxonomy_stats["rows"], 2)
            self.assertFalse(observations_zip.exists())
            self.assertFalse(taxonomy_zip.exists())
            self.assertEqual(
                json.loads(cache_path.read_text(encoding="utf-8")), {"Root": 10}
            )

            connection = duckdb.connect()
            try:
                observation = connection.execute(
                    f"SELECT * FROM '{directory / 'observations.parquet'}'"
                ).fetchone()
                taxonomy = connection.execute(
                    f"SELECT * FROM '{directory / 'taxonomy.parquet'}' ORDER BY id"
                ).fetchall()
            finally:
                connection.close()

            self.assertEqual(observation[0], 1)
            self.assertEqual(str(observation[1]), "37.123")
            self.assertEqual(str(observation[2]), "-122.988")
            self.assertEqual(taxonomy, [(10, None), (11, 10)])

    def test_incomplete_archive_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            directory = Path(temp_name)
            archive = directory / "incomplete.zip"
            archive.write_bytes(b"PK\x03\x04partial")

            with self.assertRaisesRegex(
                update_database.UpdateError, "still downloading"
            ):
                update_database.extract_member(
                    archive, "observations.csv", directory / "observations.csv"
                )


if __name__ == "__main__":
    unittest.main()
