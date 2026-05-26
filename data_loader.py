"""
Load a Kaggle CSV into Supabase with Gemini vector embeddings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED DATASET (free, no login needed to download):
  Name    : Sample Sales Data
  URL     : https://www.kaggle.com/datasets/kyanyoga/sample-sales-data
  File    : sales_data_sample.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE:
  python data_loader.py --csv sales_data_sample.csv

WHAT IT DOES:
  1. Reads every row of the CSV
  2. Converts each row to a descriptive text string
  3. Generates a 768-dim embedding via Gemini text-embedding-004
  4. Inserts content + embedding into Supabase company_data table
"""

import argparse
import os
import time

import pandas as pd
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 10     # rows between progress prints
RATE_PAUSE = 1.0    # seconds to sleep between batches (avoid API rate limits)


def row_to_text(row: pd.Series) -> str:
    """Turn one CSV row into a readable sentence-like string."""
    return " | ".join(
        f"{col}: {val}"
        for col, val in row.items()
        if pd.notna(val) and str(val).strip()
    )


def load(csv_path: str):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    gemini_key   = os.getenv("GEMINI_API_KEY")

    if not supabase_url or not supabase_key:
        print("❌  Missing SUPABASE_URL or SUPABASE_KEY in .env")
        return
    if not gemini_key:
        print("❌  Missing GEMINI_API_KEY in .env")
        return

    sb = create_client(supabase_url, supabase_key)
    genai.configure(api_key=gemini_key)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
    except FileNotFoundError:
        print(f"❌  File not found: {csv_path}")
        return

    print(f"✅  CSV loaded  →  {len(df)} rows, {len(df.columns)} columns")
    print(f"    Columns: {list(df.columns)}\n")

    ok = fail = 0

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            text = row_to_text(row)

            # Generate embedding
            emb = genai.embed_content(
                model="models/text-embedding-004",
                content=text[:8000],
            )["embedding"]

            # Insert into Supabase
            sb.table("company_data").insert({
                "content":   text,
                "metadata":  {k: str(v) for k, v in row.items() if pd.notna(v)},
                "embedding": emb,
            }).execute()

            ok += 1

        except Exception as e:
            fail += 1
            print(f"  ⚠️  Row {i} skipped — {e}")
            if fail > 20:
                print("  Too many errors, stopping.")
                break

        if (i + 1) % BATCH_SIZE == 0:
            print(f"  → {i + 1}/{len(df)} rows inserted …")
            time.sleep(RATE_PAUSE)   # be gentle with API rate limits

    print(f"\n🎉  Finished!  Inserted: {ok}  |  Failed: {fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CSV into Supabase for RAG")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    args = parser.parse_args()
    load(args.csv)
