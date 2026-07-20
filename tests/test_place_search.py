"""Tests for map-dialog place searching and viewport fitting."""

import logging
import unittest
from unittest.mock import MagicMock, call, patch

from PyQt6.QtCore import QCoreApplication
from requests import Response
from requests.exceptions import HTTPError

from visualizer import (
    PlaceSearchWorker,
    normalize_place_search_results,
    place_result_view_limits,
    place_search_response_for_log,
)


def place_response(location="37.8483,-119.5570"):
    return {
        "results": [
            {
                "type": "Place",
                "record": {
                    "id": 68542,
                    "display_name": "Yosemite National Park, US, CA",
                    "location": location,
                    "bounding_box_geojson": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-119.9, 37.4],
                                [-119.9, 38.2],
                                [-119.2, 38.2],
                                [-119.2, 37.4],
                                [-119.9, 37.4],
                            ]
                        ],
                    },
                },
            }
        ]
    }


class PlaceSearchParsingTests(unittest.TestCase):
    def test_normalizes_location_and_geojson_bounds(self) -> None:
        places = normalize_place_search_results(place_response())

        self.assertEqual(len(places), 1)
        self.assertEqual(places[0]["id"], 68542)
        self.assertAlmostEqual(places[0]["lat"], 37.8483)
        self.assertAlmostEqual(places[0]["lon"], -119.5570)
        for actual, expected in zip(
            places[0]["bounds"], (-119.9, -119.2, 37.4, 38.2), strict=True
        ):
            self.assertAlmostEqual(actual, expected)

    def test_accepts_pyinaturalist_converted_location_list(self) -> None:
        places = normalize_place_search_results(place_response([37.8483, -119.557]))

        self.assertEqual((places[0]["lat"], places[0]["lon"]), (37.8483, -119.557))

    def test_unwraps_country_bounds_around_place_center(self) -> None:
        response = place_response("45.96,-113.27")
        response["results"][0]["record"]["bounding_box_geojson"]["coordinates"] = [
            [
                [172.35, 18.87],
                [172.35, 71.44],
                [-66.89, 71.44],
                [-66.89, 18.87],
                [172.35, 18.87],
            ]
        ]

        place = normalize_place_search_results(response)[0]

        self.assertAlmostEqual(place["bounds"][0], -187.65)
        self.assertAlmostEqual(place["bounds"][1], -66.89)
        west, east, _south, _north = place_result_view_limits(place)
        self.assertLess(west, place["lon"])
        self.assertGreater(east, place["lon"])

    def test_ignores_non_places_and_invalid_coordinates(self) -> None:
        response = place_response("not,a-coordinate")
        response["results"].append({"type": "Taxon", "record": {"id": 1}})

        self.assertEqual(normalize_place_search_results(response), [])

    def test_view_limits_pad_result_bounds(self) -> None:
        limits = place_result_view_limits(
            {"lat": 37.5, "lon": -119.5, "bounds": (-120.0, -119.0, 37.0, 38.0)}
        )

        self.assertEqual(limits, (-120.1, -118.9, 36.9, 38.1))

    def test_view_limits_have_context_when_bounds_are_missing(self) -> None:
        limits = place_result_view_limits({"lat": 48.0, "lon": 2.0, "bounds": None})

        self.assertEqual(limits, (1.7, 2.3, 47.7, 48.3))

    def test_debug_response_summary_excludes_names_and_geometry(self) -> None:
        response = place_response()
        response["total_results"] = 1

        logged_response = place_search_response_for_log(response)

        self.assertIn('"total_results":1', logged_response)
        self.assertIn('"usable_places":1', logged_response)
        self.assertIn('"records_with_bounds":1', logged_response)
        self.assertNotIn("Yosemite", logged_response)
        self.assertNotIn("-119.9", logged_response)
        self.assertLess(len(logged_response), 300)


class PlaceSearchWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_worker_uses_cached_iNaturalist_place_search(self) -> None:
        request_session = MagicMock()
        emitted_results = []
        worker = PlaceSearchWorker("Yosemite", "/runtime/inat_api_cache.db")
        worker.results_ready.connect(emitted_results.append)

        with (
            patch(
                "visualizer.pyinaturalist.ClientSession",
                return_value=request_session,
            ) as client_session,
            patch(
                "visualizer.pyinaturalist.search",
                return_value=place_response([37.8483, -119.557]),
            ) as search,
        ):
            worker.run()

        client_session.assert_called_once_with(
            cache_file="/runtime/inat_api_cache.db",
            max_retries=0,
            timeout=8,
        )
        search.assert_called_once_with(
            q="Yosemite",
            sources="places",
            per_page=8,
            session=request_session,
        )
        self.assertEqual(emitted_results[0][0]["id"], 68542)
        request_session.close.assert_called_once_with()

    def test_worker_resolves_qualifier_and_logs_safe_summaries(self) -> None:
        request_session = MagicMock()
        emitted_results = []
        exact_response = {"total_results": 0, "results": []}
        qualifier_response = place_response([-9.19, -75.02])
        qualifier_record = qualifier_response["results"][0]["record"]
        qualifier_record["id"] = 7513
        qualifier_record["name"] = "Peru"
        qualifier_record["display_name"] = "Peru"
        qualifier_record["admin_level"] = 0
        fallback_response = place_response([-10.58, -75.40])
        fallback_record = fallback_response["results"][0]["record"]
        fallback_record["id"] = 40827
        fallback_record["name"] = "Oxapampa"
        fallback_record["display_name"] = "Oxapampa, PA, PE"
        fallback_record["ancestor_place_ids"] = [97389, 7513, 11538, 40827]
        other_result = place_response([40.0, -75.0])["results"][0]
        other_result["record"]["id"] = 99999
        other_result["record"]["name"] = "Oxapampa"
        other_result["record"]["display_name"] = "Oxapampa, Somewhere Else"
        other_result["record"]["ancestor_place_ids"] = [1, 2, 99999]
        fallback_response["results"].append(other_result)
        worker = PlaceSearchWorker("Oxapampa, Peru")
        worker.results_ready.connect(emitted_results.append)

        with (
            patch(
                "visualizer.pyinaturalist.ClientSession",
                return_value=request_session,
            ),
            patch(
                "visualizer.pyinaturalist.search",
                side_effect=[exact_response, qualifier_response, fallback_response],
            ) as search,
            self.assertLogs(level=logging.DEBUG) as captured_logs,
        ):
            worker.run()

        self.assertEqual(
            search.call_args_list,
            [
                call(
                    session=request_session,
                    q="Oxapampa, Peru",
                    sources="places",
                    per_page=8,
                ),
                call(
                    session=request_session,
                    q="Peru",
                    sources="places",
                    per_page=8,
                ),
                call(
                    session=request_session,
                    q="Oxapampa",
                    sources="places",
                    per_page=8,
                ),
            ],
        )
        log_output = "\n".join(captured_logs.output)
        self.assertEqual(log_output.count("Place search completed:"), 3)
        self.assertIn('"total_results":0', log_output)
        self.assertIn("Place search fallback:", log_output)
        self.assertIn("kept 1 of 2 fallback matches", log_output)
        self.assertNotIn("Oxapampa", log_output)
        self.assertNotIn("Peru", log_output)
        self.assertNotIn("Somewhere Else", log_output)
        self.assertEqual(emitted_results[0][0]["display_name"], "Oxapampa, PA, PE")

    def test_worker_reports_search_failures(self) -> None:
        errors = []
        worker = PlaceSearchWorker("Nowhere")
        worker.search_failed.connect(errors.append)

        with self.assertLogs(level=logging.ERROR):
            with (
                patch(
                    "visualizer.pyinaturalist.ClientSession",
                    return_value=MagicMock(),
                ),
                patch(
                    "visualizer.pyinaturalist.search",
                    side_effect=RuntimeError("offline"),
                ),
            ):
                worker.run()

        self.assertEqual(errors, ["offline"])

    def test_worker_logs_safe_http_exception_chain_diagnostics(self) -> None:
        response = Response()
        response.status_code = 503
        response.url = "https://api.inaturalist.org/v1/search?q=PrivatePlace"
        http_error = HTTPError("request for PrivatePlace failed", response=response)
        try:
            raise RuntimeError("wrapper mentions PrivatePlace") from http_error
        except RuntimeError as error:
            chained_error = error

        worker = PlaceSearchWorker("PrivatePlace")
        with (
            patch(
                "visualizer.pyinaturalist.ClientSession",
                return_value=MagicMock(),
            ),
            patch(
                "visualizer.pyinaturalist.search",
                side_effect=chained_error,
            ),
            self.assertLogs(level=logging.ERROR) as captured_logs,
        ):
            worker.run()

        output = "\n".join(captured_logs.output)
        self.assertIn("error_chain=RuntimeError -> HTTPError", output)
        self.assertIn("http_status=503", output)
        self.assertIn("response_host=api.inaturalist.org", output)
        self.assertNotIn("PrivatePlace", output)


if __name__ == "__main__":
    unittest.main()
