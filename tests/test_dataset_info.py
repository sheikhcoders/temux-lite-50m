"""Unit tests for the dataset inspection helpers."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("requests")

from scripts import dataset_info


def test_build_rows_url_encodes_parts() -> None:
    url = dataset_info.build_rows_url("GetSoloTech/Code-Reasoning", "default", "python", 0, 100)
    assert (
        url
        == "https://datasets-server.huggingface.co/rows?dataset=GetSoloTech%2FCode-Reasoning&config=default&split=python&offset=0&length=100"
    )


def test_build_parquet_url() -> None:
    url = dataset_info.build_parquet_url("GetSoloTech/Code-Reasoning", "default", "python")
    assert url == "https://huggingface.co/api/datasets/GetSoloTech%2FCode-Reasoning/parquet/default/python"


def test_fetch_rows_success() -> None:
    response = mock.Mock()
    response.json.return_value = {"rows": [1, 2, 3]}
    response.raise_for_status.return_value = None
    with mock.patch.object(dataset_info.requests, "get", return_value=response) as mock_get:
        payload = dataset_info.fetch_rows("ds", None, "train", 0, 10)
    mock_get.assert_called_once()
    assert payload == {"rows": [1, 2, 3]}


def test_fetch_rows_error() -> None:
    response = mock.Mock()
    response.raise_for_status.side_effect = dataset_info.requests.HTTPError("boom")
    with mock.patch.object(dataset_info.requests, "get", return_value=response):
        with pytest.raises(dataset_info.DatasetInfoError):
            dataset_info.fetch_rows("ds", None, "train", 0, 10)
