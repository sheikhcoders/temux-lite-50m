"""Utility helpers for inspecting Hugging Face datasets server metadata."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.parse import quote

import requests

DATASETS_SERVER = "https://datasets-server.huggingface.co"
HUB_API = "https://huggingface.co/api"


class DatasetInfoError(RuntimeError):
    """Raised when the datasets server returns an unexpected response."""


def _request_json(url: str) -> Any:
    response = requests.get(url, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - exercised in CLI usage
        raise DatasetInfoError(f"Request failed for {url}: {exc}") from exc
    return response.json()


def build_rows_url(dataset: str, config: str | None, split: str, offset: int, length: int) -> str:
    base = f"{DATASETS_SERVER}/rows?dataset={quote(dataset, safe='')}"
    if config:
        base += f"&config={quote(config, safe='')}"
    base += f"&split={quote(split, safe='')}&offset={offset}&length={length}"
    return base


def fetch_rows(dataset: str, config: str | None, split: str, offset: int, length: int) -> Any:
    url = build_rows_url(dataset, config, split, offset, length)
    return _request_json(url)


def build_splits_url(dataset: str) -> str:
    return f"{DATASETS_SERVER}/splits?dataset={quote(dataset, safe='')}"


def fetch_splits(dataset: str) -> Any:
    return _request_json(build_splits_url(dataset))


def build_parquet_url(dataset: str, config: str, split: str) -> str:
    return f"{HUB_API}/datasets/{quote(dataset, safe='')}/parquet/{quote(config, safe='')}/{quote(split, safe='')}"


def fetch_parquet_files(dataset: str, config: str, split: str) -> Any:
    return _request_json(build_parquet_url(dataset, config, split))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset repository id, e.g. 'GetSoloTech/Code-Reasoning'")
    parser.add_argument("--config", default="default", help="Dataset config name")
    parser.add_argument("--split", default="python", help="Dataset split to inspect")
    parser.add_argument("--offset", type=int, default=0, help="Row offset for sample fetches")
    parser.add_argument("--length", type=int, default=100, help="Number of rows to request")
    parser.add_argument(
        "--show-parquet",
        action="store_true",
        help="List parquet files in addition to rows and split metadata",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON responses")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    rows_payload = fetch_rows(args.dataset, args.config, args.split, args.offset, args.length)
    splits_payload = fetch_splits(args.dataset)

    parquet_payload = None
    if args.show_parquet:
        parquet_payload = fetch_parquet_files(args.dataset, args.config, args.split)

    if args.pretty:
        json.dump({
            "rows": rows_payload,
            "splits": splits_payload,
            "parquet": parquet_payload,
        }, sys.stdout, indent=2)
    else:
        print(json.dumps({"rows": rows_payload, "splits": splits_payload, "parquet": parquet_payload}))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
