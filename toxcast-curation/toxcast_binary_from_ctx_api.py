from __future__ import annotations

import os
import re
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


@dataclass
class Config:

    bioactivity_base: str = "https://comptox.epa.gov/ctx-api/bioactivity"
    chemical_base: str = "https://comptox.epa.gov/ctx-api/chemical"

    api_key: Optional[str] = os.getenv("CTX_API_KEY")


    active_threshold: float = 0.90

    # --- Floating-point tolerance (for handling cases like 1+1e-12) ---
    eps: float = 1e-6

    # Strategy for handling hitc < 0
    # "na"      : hitc<0 -> NA (not applicable / ignored)
    # "inactive": hitc<0 -> treat as inactive (0)
    # "abs"     : hitc <- abs(hitc) (only recommended if you don't want to preserve directional info)
    negative_hitc_policy: str = "na"

    timeout: int = 120

    request_sleep: float = 0.05

    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    out_dir: str = "toxcast_out"


    max_assays: Optional[int] = 20

    smiles_workers: int = 8

    chemical_projection: str = "chemicalstructure"

    save_parquet: bool = True
    save_csv: bool = False


def _headers(cfg: Config) -> Dict[str, str]:
    """
    Use the 'x-api-key' header as ctxR does.
    If cfg.api_key is None, the header is omitted (some endpoints may still be accessible).
    """
    h = {"Content-Type": "application/json"}
    if cfg.api_key:
        h["x-api-key"] = cfg.api_key
    return h


def create_session(cfg: Config) -> requests.Session:
    """
    Create a requests.Session configured with a retry strategy.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=cfg.max_retries,
        backoff_factor=cfg.retry_backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def get_json(cfg: Config, url: str, params: Optional[Dict[str, Any]] = None, session: Optional[requests.Session] = None) -> Any:
    """
    Send a GET request and return the JSON response.
    - cfg.timeout: prevents hanging
    - cfg.request_sleep: throttling
    - session: optional session object for connection reuse
    """
    if session is None:
        session = create_session(cfg)

    try:
        r = session.get(url, headers=_headers(cfg), params=params, timeout=cfg.timeout)
        if r.status_code == 401:
            raise RuntimeError("HTTP 401 Unauthorized: a valid x-api-key (CTX_API_KEY) may be required")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} error for {url}: {r.text[:500]}")
        time.sleep(cfg.request_sleep)
        return r.json()
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Request timed out (timeout={cfg.timeout}s): {url}") from e
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Connection error: {url}") from e


# =========================
# C) Bioactivity: fetch assays and AEID-associated data
# =========================
def fetch_all_assays(cfg: Config, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    GET {bioactivity_base}/assay/ to retrieve annotations for all assays (AEIDs).
    The current API returns a list, which is directly converted to a DataFrame.
    """
    url = f"{cfg.bioactivity_base}/assay/"
    data = get_json(cfg, url, session=session)

    if not isinstance(data, list):
        raise TypeError(f"/assay/ expected list, got {type(data)}")

    df = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    return df



def fetch_bioactivity_by_aeid(cfg: Config, aeid: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Endpoint shown in ctxR: GET {Server}/data/search/by-aeid/{AEID}
    Typically returns the processed results (including hitc) for each chemical in the assay.
    """
    url = f"{cfg.bioactivity_base}/data/search/by-aeid/{int(aeid)}"

    data = get_json(cfg, url, session=session)
    if not isinstance(data, list):
        raise TypeError(f"/data/search/by-aeid/{aeid} expected list, got {type(data)}")

    df = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    return df    

def clean_hitc_value(cfg: Config, x: Any) -> Optional[float]:
    """
    Clean hitc into a float suitable for binarization:
    1) Cannot convert to float -> NA
    2) In range (-eps, 0) -> treat as 0 (floating-point jitter)
    3) In range (1, 1+eps) -> treat as 1 (floating-point jitter)
    4) x < 0 (and less than -eps) -> handled by negative_hitc_policy
    5) x > 1+eps -> treated as anomalous, clipped to 1 (or change to drop/raise)
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None

    try:
        v = float(x)
    except Exception:
        return None

    # Very small negative: treat as floating-point error, snap to 0
    if -cfg.eps <= v < 0:
        v = 0.0

    # Very slightly above 1: treat as floating-point error, snap to 1
    if 1 < v <= 1 + cfg.eps:
        v = 1.0

    if v < 0:
        if cfg.negative_hitc_policy == "na":
            return None
        if cfg.negative_hitc_policy == "inactive":
            return 0.0
        if cfg.negative_hitc_policy == "abs":
            v = abs(v)
        else:
            raise ValueError("negative_hitc_policy must be one of: na, inactive, abs")

    if v > 1:
        v = 1.0

    return v


def binarize_hitc(cfg: Config, hitc: Optional[float]) -> Optional[int]:
    """
    hitc -> binary active label
    - hitc is None or NaN -> None
    - hitc >= threshold -> 1
    - else -> 0
    """

    if hitc is None or (isinstance(hitc, float) and math.isnan(hitc)):

        return None
    return int(hitc >= cfg.active_threshold)


def extract_smiles_from_detail(detail_json: Any) -> Optional[str]:
    """
    Fields in the chemical detail response may vary by projection.
    We extract by searching for keys containing 'smiles', prioritizing canonical/preferred variants.
    """
    if detail_json is None:
        return None

    candidates: List[Dict[str, Any]] = []
    if isinstance(detail_json, dict):
        candidates = [detail_json]
    elif isinstance(detail_json, list) and len(detail_json) > 0:
        # Some APIs return a list directly
        candidates = [d for d in detail_json if isinstance(d, dict)]

    best = None
    best_score = -1

    for d in candidates:
        for k, v in d.items():
            if not isinstance(k, str):
                continue
            lk = k.lower()
            if "smiles" not in lk:
                continue
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue

            # Score: higher score for canonical/preferred variants
            score = 0
            if "canon" in lk:
                score += 3
            if "preferred" in lk:
                score += 2
            if lk == "smiles":
                score += 1

            if score > best_score:
                best = s
                best_score = score

    return best


def fetch_smiles_for_dtxsid(cfg: Config, dtxsid: str, session: Optional[requests.Session] = None) -> Tuple[str, Optional[str]]:
    """
    Chemical details endpoint from ctxR:
    GET {Server}/detail/search/by-dtxsid/{DTXSID}?projection=...
    """
    dtxsid = str(dtxsid)
    url = f"{cfg.chemical_base}/detail/search/by-dtxsid/{dtxsid}"
    params = {"projection": cfg.chemical_projection}
    try:
        detail = get_json(cfg, url, params=params, session=session)
        smiles = extract_smiles_from_detail(detail)
        return dtxsid, smiles
    except Exception as e:
        print(f"[WARN] Failed to fetch SMILES for {dtxsid}: {type(e).__name__}: {str(e)}")
        return dtxsid, None

def ensure_outdir(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "by_aeid"), exist_ok=True)


def aeid_output_path(cfg: Config, aeid: int) -> str:
    # Each AEID is written to its own file for easy resume after interruption
    ext = "parquet" if cfg.save_parquet else "csv"
    return os.path.join(cfg.out_dir, "by_aeid", f"aeid_{int(aeid)}.{ext}")


def save_df(cfg: Config, df: pd.DataFrame, path: str) -> None:
    if cfg.save_parquet:
        df.to_parquet(path, index=False)
    elif cfg.save_csv:
        df.to_csv(path, index=False)
    else:
        raise ValueError("At least one of save_parquet or save_csv must be enabled")


def load_df(path: str) -> pd.DataFrame:
    """
    Load a DataFrame, handling empty or corrupted files.
    """
    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            # Check file size; if too small it may be empty
            if os.path.getsize(path) < 10:
                # Return an empty DataFrame with the correct schema
                return pd.DataFrame(columns=["dtxsid", "aeid", "hitc_clean", "active"])
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unknown file type: {path}")

        # Ensure required columns exist
        if df.empty or len(df.columns) == 0:
            return pd.DataFrame(columns=["dtxsid", "aeid", "hitc_clean", "active"])

        return df
    except (pd.errors.EmptyDataError, Exception) as e:
        print(f"[WARN] Cannot read file {path}: {type(e).__name__}: {str(e)}")
        # Return an empty but correctly structured DataFrame
        return pd.DataFrame(columns=["dtxsid", "aeid", "hitc_clean", "active"])


def process_one_aeid(cfg: Config, aeid: int, overwrite: bool = False, session: Optional[requests.Session] = None) -> Optional[str]:
    """
    Download and clean a single AEID, write output file, and return its path.
    Errors are logged and None is returned rather than interrupting the full pipeline.
    """
    out_path = aeid_output_path(cfg, aeid)
    if (not overwrite) and os.path.exists(out_path):
        return out_path

    try:
        df = fetch_bioactivity_by_aeid(cfg, aeid, session=session)
    except Exception as e:
        print(f"[ERROR] AEID={aeid} fetch failed: {type(e).__name__}: {str(e)}")
        return None

    if df.empty:
        # Still write an empty file (with correct schema) to avoid retrying indefinitely
        empty_df = pd.DataFrame(columns=["dtxsid", "aeid", "hitc_clean", "active"])
        empty_df["aeid"] = empty_df["aeid"].astype('Int64')  # Use nullable int
        save_df(cfg, empty_df, out_path)
        return out_path

    # Extract core columns: dtxsid, aeid, hitc
    # Common column names: dtxsid, aeid, hitc (all lowercased already)
    if "dtxsid" not in df.columns:
        # Some data may use dtxsid_x or similar; find the closest match
        dtx_cols = [c for c in df.columns if "dtxsid" in c]
        if dtx_cols:
            df = df.rename(columns={dtx_cols[0]: "dtxsid"})

    if "aeid" not in df.columns:
        df["aeid"] = int(aeid)

    if "hitc" not in df.columns:
        print(f"[WARN] AEID={aeid} skipped: missing required column 'hitc'. "
              f"Available columns: {list(df.columns)[:30]} ...")
        return None

    # Keep only required columns (add others back if needed)
    keep_cols = [c for c in ["dtxsid", "aeid", "hitc"] if c in df.columns]
    df2 = df[keep_cols].copy()

    # Clean and binarize
    df2["hitc_clean"] = df2["hitc"].apply(lambda x: clean_hitc_value(cfg, x))
    df2["active"] = df2["hitc_clean"].apply(lambda v: binarize_hitc(cfg, v))

    # Aggregate duplicate (DTXSID, AEID) entries using max(hitc_clean) (can change to mean/median)
    df2 = (
        df2.groupby(["dtxsid", "aeid"], as_index=False)
           .agg(hitc_clean=("hitc_clean", "max"),
                active=("active", "max"))
    )

    try:
        save_df(cfg, df2, out_path)
        return out_path
    except Exception as e:
        print(f"[ERROR] AEID={aeid} save failed: {type(e).__name__}: {str(e)}")
        return None


def combine_all_aeids(cfg: Config, paths: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Concatenate per-AEID long tables into a single large long table.
    Returns: (combined DataFrame, list of AEIDs with empty files)
    """
    frames = []
    empty_aeids = []
    error_aeids = []

    for p in tqdm(paths, desc="Combining AEID files"):
        # Extract AEID number from filename
        match = re.search(r'aeid_(\d+)\.(csv|parquet)', os.path.basename(p))
        aeid = match.group(1) if match else "unknown"

        try:
            df = load_df(p)
            if not df.empty:
                frames.append(df)
            else:
                empty_aeids.append(aeid)
        except Exception as e:
            print(f"[ERROR] Cannot load {p}: {type(e).__name__}: {str(e)}")
            error_aeids.append(aeid)

    # Summary statistics
    print(f"\n[INFO] Merge summary: valid files {len(frames)}, empty files {len(empty_aeids)}, errors {len(error_aeids)}")

    if empty_aeids:
        print(f"[INFO] Empty AEIDs ({len(empty_aeids)}): {', '.join(empty_aeids[:30])}")
        if len(empty_aeids) > 30:
            print(f"       ... and {len(empty_aeids) - 30} more")

    if error_aeids:
        print(f"[WARN] AEIDs with read errors ({len(error_aeids)}): {', '.join(error_aeids)}")

    if not frames:
        return pd.DataFrame(columns=["dtxsid", "aeid", "hitc_clean", "active"]), empty_aeids
    return pd.concat(frames, ignore_index=True), empty_aeids


def build_smiles_table(cfg: Config, dtxsids: List[str], session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Fetch SMILES in parallel using multiple threads; returns (dtxsid, smiles).
    """
    rows = []
    with ThreadPoolExecutor(max_workers=cfg.smiles_workers) as ex:
        futs = {ex.submit(fetch_smiles_for_dtxsid, cfg, d, session): d for d in dtxsids}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Fetching SMILES"):
            dtxsid, smiles = fut.result()
            rows.append((dtxsid, smiles))
    return pd.DataFrame(rows, columns=["dtxsid", "smiles"])


def create_aeid_to_assay_name_mapping(assays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract AEID-to-assay-name mapping from the assays DataFrame.
    Common assay name columns: assay_component_endpoint_name, aenm, assay_name, etc.
    """
    # Candidate assay name columns in order of preference
    possible_name_cols = [
        "assay_component_endpoint_name", "aenm", "assay_name",
        "assaycomponentendpointname", "endpoint_name", "name"
    ]

    assay_name_col = None
    for col in possible_name_cols:
        if col in assays_df.columns:
            assay_name_col = col
            break

    if assay_name_col is None:
        # Fall back to the first column whose name contains "name"
        name_like_cols = [c for c in assays_df.columns if "name" in c.lower()]
        if name_like_cols:
            assay_name_col = name_like_cols[0]
        else:
            # Last resort: use AEID itself as the name
            print("[WARN] No assay name column found; using AEID as the name")
            mapping = assays_df[["aeid"]].copy()
            mapping["assay_name"] = "AEID_" + mapping["aeid"].astype(str)
            return mapping[["aeid", "assay_name"]]

    print(f"[INFO] Using '{assay_name_col}' as the assay name column")
    mapping = assays_df[["aeid", assay_name_col]].copy()
    mapping.columns = ["aeid", "assay_name"]

    # Fill any missing assay names
    mapping["assay_name"] = mapping["assay_name"].fillna("AEID_" + mapping["aeid"].astype(str))

    return mapping


def convert_to_wide_format(long_df: pd.DataFrame, aeid_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the long table to wide format:
    - Rows: SMILES
    - Columns: Assay Name
    - Values: active (0/1/NaN)
    """
    # Map AEID to assay name
    df_with_names = long_df.merge(aeid_mapping, on="aeid", how="left")

    # Handle AEIDs without a mapped name (should not normally occur)
    df_with_names["assay_name"] = df_with_names["assay_name"].fillna(
        "AEID_" + df_with_names["aeid"].astype(str)
    )

    # Aggregate duplicate (smiles, assay_name) pairs using max(active) (conservative)
    df_agg = (
        df_with_names.groupby(["smiles", "assay_name"], as_index=False)
        .agg(active=("active", "max"))
    )

    # Pivot to wide format
    wide_df = df_agg.pivot(index="smiles", columns="assay_name", values="active")

    # Reset index so SMILES becomes a regular column
    wide_df = wide_df.reset_index()

    print(f"[INFO] Wide table dimensions: {wide_df.shape[0]} molecules × {wide_df.shape[1]-1} assays")

    return wide_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="toxcast_out")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max_assays", type=int, default=-1)  # -1 表示全做
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--save_parquet", action="store_true")
    parser.add_argument("--negative_policy", type=str, default="na", choices=["na", "inactive", "abs"])
    args = parser.parse_args()

    cfg = Config(
        out_dir=args.out_dir,
        active_threshold=args.threshold,
        max_assays=None if args.max_assays == -1 else args.max_assays,
        negative_hitc_policy=args.negative_policy,
        save_csv=args.save_csv,
        save_parquet=(args.save_parquet or (not args.save_csv)),  # Default to parquet if neither is specified
    )
    # detail = get_json(
    #     cfg,
    #     f"{cfg.chemical_base}/detail/search/by-dtxsid/DTXSID7020182",
    #     params={"projection": cfg.chemical_projection},
    # )

    # print("\n[DEBUG] type(detail) =", type(detail))
    # if isinstance(detail, dict):
    #     print("[DEBUG] keys:", list(detail.keys())[:80])
    # elif isinstance(detail, list) and detail and isinstance(detail[0], dict):
    #     print("[DEBUG] keys0:", list(detail[0].keys())[:80])

    ensure_outdir(cfg)

    # Create a shared session to reduce connection overhead
    session = create_session(cfg)
    print(f"[INFO] Config: timeout={cfg.timeout}s, max_retries={cfg.max_retries}, backoff_factor={cfg.retry_backoff_factor}")

    # 1) Fetch all assays
    assays = fetch_all_assays(cfg, session=session)
    if assays.empty:
        raise RuntimeError("assay list is empty; check API base url / API key")

    test_aeid = int(assays["aeid"].iloc[0])
    df_one = fetch_bioactivity_by_aeid(cfg, test_aeid, session=session)

    # print("AEID =", test_aeid)
    # print("shape =", df_one.shape)
    # print("columns =", df_one.columns.tolist()[:80])
    # print(df_one[["dtxsid", "hitc"]].head() if ("dtxsid" in df_one.columns and "hitc" in df_one.columns) else df_one.head())

    if "aeid" not in assays.columns:
        # Conservatively search for the AEID column
        cols = [c for c in assays.columns if c.lower() == "aeid" or "aeid" in c.lower()]
        if not cols:
            raise RuntimeError(f"Cannot find AEID column in assay table. Columns={assays.columns.tolist()}")
        assays = assays.rename(columns={cols[0]: "aeid"})

    # Build and save the AEID → Assay Name mapping
    print("[INFO] Building AEID → Assay Name mapping...")
    aeid_mapping = create_aeid_to_assay_name_mapping(assays)
    mapping_path = os.path.join(cfg.out_dir, "aeid_to_assay_name.csv")
    aeid_mapping.to_csv(mapping_path, index=False)
    print(f"[INFO] AEID mapping saved to: {mapping_path}")

    aeids = assays["aeid"].dropna().astype(int).tolist()
    if cfg.max_assays is not None:
        aeids = aeids[: cfg.max_assays]

    # 2) Per-AEID download + clean (with resume support)
    already_completed = []
    if not args.overwrite:
        for aeid in aeids:
            if os.path.exists(aeid_output_path(cfg, aeid)):
                already_completed.append(aeid)

        if already_completed:
            completed_count = len(already_completed)
            first_pending = next((aeid for aeid in aeids if aeid not in already_completed), None)
            if first_pending:
                print(f"[INFO] Resuming from AEID={first_pending}...")
            else:
                print("[INFO] All AEIDs already completed; proceeding to merge step")

    paths = []
    failed_aeids = []
    with tqdm(total=len(aeids), desc="Processing AEIDs", initial=len(already_completed)) as pbar:
        for aeid in aeids:
            p = process_one_aeid(cfg, aeid, overwrite=args.overwrite, session=session)
            if p:
                paths.append(p)
            else:
                failed_aeids.append(aeid)

            # Only update progress bar for newly processed AEIDs
            if aeid not in already_completed:
                pbar.update(1)

    print(f"\n[INFO] Processing complete: {len(paths)}/{len(aeids)} AEIDs succeeded")
    if failed_aeids:
        print(f"[WARN] Failed AEIDs ({len(failed_aeids)}): {failed_aeids[:20]}{'...' if len(failed_aeids) > 20 else ''}")
        # Write failed AEIDs to file for later retry
        failed_log = os.path.join(cfg.out_dir, "failed_aeids.txt")
        with open(failed_log, "w") as f:
            for aeid in failed_aeids:
                f.write(f"{aeid}\n")
        print(f"[INFO] Failed AEID list saved to: {failed_log}")

    # 3) Combine per-AEID files
    long_df, empty_aeids = combine_all_aeids(cfg, paths)

    # Save list of empty AEIDs
    if empty_aeids:
        empty_log = os.path.join(cfg.out_dir, "empty_aeids.txt")
        with open(empty_log, "w") as f:
            f.write("# The following AEIDs returned no data from the API\n")
            for aeid in sorted(empty_aeids, key=lambda x: int(x) if x.isdigit() else 0):
                f.write(f"{aeid}\n")
        print(f"[INFO] Empty AEID list saved to: {empty_log}")

    # 4) Build SMILES table
    dtxsids = sorted([d for d in long_df["dtxsid"].dropna().astype(str).unique().tolist() if d])
    print(f"[INFO] Fetching SMILES for {len(dtxsids)} DTXSIDs...")
    smiles_df = build_smiles_table(cfg, dtxsids, session=session)

    smiles_path = os.path.join(cfg.out_dir, "dtxsid_smiles.parquet")
    smiles_csv = os.path.join(cfg.out_dir, "dtxsid_smiles.csv")
    if cfg.save_parquet:
        smiles_df.to_parquet(smiles_path, index=False)
    if cfg.save_csv:
        smiles_df.to_csv(smiles_csv, index=False)

    # 5) Merge SMILES into long format (for intermediate processing)
    merged = long_df.merge(smiles_df, on="dtxsid", how="left")

    # Drop rows without SMILES
    merged = merged[merged["smiles"].notna()].copy()
    print(f"[INFO] Valid records (with SMILES): {len(merged)}")

    # 6) Convert to wide format
    wide_df = convert_to_wide_format(merged, aeid_mapping)

    # 7) Save wide table
    wide_path = os.path.join(cfg.out_dir, "toxcast_binary_wide.parquet")
    wide_csv = os.path.join(cfg.out_dir, "toxcast_binary_wide.csv")

    if cfg.save_parquet:
        wide_df.to_parquet(wide_path, index=False)
        print(f"[INFO] Wide table (parquet) saved to: {wide_path}")
    if cfg.save_csv:
        wide_df.to_csv(wide_csv, index=False)
        print(f"[INFO] Wide table (csv) saved to: {wide_csv}")

    print(f"- AEID mapping: {mapping_path}")
    print(f"- SMILES mapping: {smiles_path if cfg.save_parquet else smiles_csv}")
    print(f"- Wide table (SMILES × Assay Name): {wide_path if cfg.save_parquet else wide_csv}")
    print(f"  Dimensions: {wide_df.shape[0]} molecules × {wide_df.shape[1]-1} assays")


if __name__ == "__main__":
    main()
