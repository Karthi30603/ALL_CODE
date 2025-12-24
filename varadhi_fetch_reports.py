#!/usr/bin/env python3
"""
Utility to fetch report documents from the Varadhi Typesense cluster
for a list of study_iuid identifiers and persist them to a CSV file.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm
import typesense  # type: ignore


DEFAULT_STUDY_IUIDS: List[str] = [
    "1.2.840.113619.2.482.3.2831215105.864.1763872364.97",
    "1.3.12.2.1107.5.1.7.107063.30000025111422064655000000476",
    "2.25.799113963749809659915162928890391383165",
    "1.2.840.113619.2.415.3.2831198978.165.1763913609.521",
    "1.2.156.112605.189250954715127.251123155700.2.12184.13750",
    "1.2.392.200036.9116.2.6.1.48.1214243189.1763906622.527274",
    "1.3.46.670589.33.1.63899512394400137600001.5647284570191405197",
    "1.2.840.113619.2.81.325.1.14706.20251123.292213",
    "1.2.840.113619.2.415.3.2831166721.864.1763870206.950",
    "1.2.840.113619.2.411.3.380401623.144.1763874557.835",
    "1.2.840.113704.9.1000.16.0.20251123200013390",
    "2.25.289009986494790467630156409695883630067",
    "1.2.840.113619.2.417.3.2831165771.798.1763560711.216",
    "1.2.840.113619.2.438.3.2071659131.50.1763866653.50",
    "2.25.497391741694139865845022814205607844452",
    "2.25.696651404244294706108840783197695984596",
    "1.2.840.113619.2.415.3.2831179521.238.1763897680.963",
    "1.3.12.2.1107.5.1.7.108180.30000025112314134131800000014",
    "2.25.956470794561439181587080488647110511376",
]


def get_typesense_client() -> typesense.Client:
    """Instantiate a Typesense client, allowing host/key overrides via env."""
    host = os.getenv("TYPESENSE_HOST", "api-varadhi.cubebase.ai")
    api_key = os.getenv("TYPESENSE_API_KEY", "0febb6ff-4691-4ed1-a5ce-28e9ae2ae452")
    timeout = int(os.getenv("TYPESENSE_TIMEOUT", "20"))

    return typesense.Client(
        {
            "nodes": [{"host": host, "port": "443", "protocol": "https"}],
            "api_key": api_key,
            "connection_timeout_seconds": timeout,
        }
    )


def fetch_by_study_iuid(client: typesense.Client, study_iuid: str) -> list[dict]:
    """Fetch hits for a single study_iuid."""
    search_parameters = {
        "q": "*",
        "query_by": "study_iuid",
        "filter": f'study_iuid:="{study_iuid}"',
        "sort_by": "created_at:desc",
    }
    search_results = client.collections["report_contents"].documents.search(search_parameters)
    return search_results.get("hits", [])


def load_study_iuids_from_csv(csv_path: Path, column: str | None, limit: int | None) -> List[str]:
    """Load study_iuid values from a CSV file, auto-detecting the column when needed."""
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    candidate_cols = ["study_iuid", "s.study_iuid"]
    column_name = column or next((col for col in candidate_cols if col in df.columns), None)
    if column_name is None:
        raise ValueError(
            f"Unable to find a study_iuid column. Available columns: {list(df.columns)}"
        )
    return df[column_name].astype(str).tolist()


def fetch_with_retry(
    client: typesense.Client, study_iuid: str, retries: int, backoff_seconds: float
) -> list[dict]:
    """Fetch with simple linear backoff retries."""
    attempt = 0
    while True:
        try:
            return fetch_by_study_iuid(client, study_iuid)
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise exc
            sleep_for = backoff_seconds * attempt
            print(f"Retrying {study_iuid} in {sleep_for:.1f}s after error: {exc}")
            time.sleep(sleep_for)


def gather_documents(study_iuids: Iterable[str], retries: int, backoff_seconds: float) -> list[dict]:
    """Fetch documents for each study_iuid and return the combined hits."""
    client = get_typesense_client()
    hits: list[dict] = []
    for study_iuid in tqdm(study_iuids, desc="Fetching documents"):
        try:
            results = fetch_with_retry(client, study_iuid, retries=retries, backoff_seconds=backoff_seconds)
            hits.extend(results)
        except Exception as exc:
            print(f"Error fetching study_iuid {study_iuid}: {exc}")
    return hits


def save_hits_to_csv(hits: list[dict], output_path: Path) -> None:
    """Persist the raw document payloads to CSV."""
    documents = [hit["document"] for hit in hits if "document" in hit]
    if not documents:
        print("No documents retrieved; skipping CSV export.")
        return
    df = pd.DataFrame(documents)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(documents)} documents to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch report_contents documents for study_iuid identifiers."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV containing study_iuid values.",
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Column name to use from the CSV (auto-detected if omitted).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of rows read from CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("varadhi_exact_search_results.csv"),
        help="Location for the output CSV.",
    )
    parser.add_argument(
        "--study-iuid",
        dest="study_iuid_overrides",
        action="append",
        help="Explicit study_iuid values (can be repeated).",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries per study_iuid on failures."
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Seconds to wait between retries (multiplied by attempt number).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        study_iuids = load_study_iuids_from_csv(args.csv, args.column, args.limit)
    elif args.study_iuid_overrides:
        study_iuids = args.study_iuid_overrides
    else:
        study_iuids = DEFAULT_STUDY_IUIDS

    hits = gather_documents(study_iuids, retries=args.retries, backoff_seconds=args.backoff)
    if not hits:
        print("⚠️ No results found for the provided study IDs.")
        return

    save_hits_to_csv(hits, args.output)


if __name__ == "__main__":
    main()

