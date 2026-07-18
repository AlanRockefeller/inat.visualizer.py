import unittest
from unittest.mock import MagicMock, patch

from visualizer import fetch_observations_page, should_post_observation_search


class TestObservationRequestFallback(unittest.TestCase):
    def test_large_taxon_filters_switch_to_post(self):
        params = {
            "lat": 37.7749,
            "lng": -122.4194,
            "radius": 25,
            "d1": "2025-01-01",
            "d2": "2025-12-31",
            "taxon_id": list(range(100000, 100220)),
            "page": 1,
            "per_page": 500,
        }

        self.assertTrue(should_post_observation_search(params))

        response = MagicMock()
        response.json.return_value = {"results": [{"id": 1}], "total_results": 1}
        response.raise_for_status.return_value = None

        with patch("visualizer.requests.post", return_value=response) as post_mock:
            with patch("visualizer.pyinaturalist.get_observations") as get_mock:
                result = fetch_observations_page(params)

        get_mock.assert_not_called()
        post_mock.assert_called_once()
        self.assertEqual(result["results"][0]["id"], 1)

    def test_small_taxon_filters_stay_on_get(self):
        params = {
            "lat": 37.7749,
            "lng": -122.4194,
            "radius": 25,
            "d1": "2025-01-01",
            "d2": "2025-12-31",
            "taxon_id": [47125, 20978],
            "page": 1,
            "per_page": 500,
        }

        self.assertFalse(should_post_observation_search(params))

        with patch(
            "visualizer.pyinaturalist.get_observations",
            return_value={"results": [{"id": 2}], "total_results": 1},
        ) as get_mock:
            with patch("visualizer.requests.post") as post_mock:
                result = fetch_observations_page(params)

        post_mock.assert_not_called()
        get_mock.assert_called_once_with(**params)
        self.assertEqual(result["results"][0]["id"], 2)
