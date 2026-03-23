# ToxCast Curation — Usage Guide

`toxcast_binary_from_ctx_api.py` fetches ToxCast bioactivity data from the [CompTox CTX API](https://comptox.epa.gov/ctx-api/), binarizes `hitc` values, retrieves SMILES for each compound, and outputs a wide-format table of **molecules × assays**.

---

## Prerequisites

An API key (`CTX_API_KEY`) need be required

```bash
export CTX_API_KEY="your_api_key_here"
```

---

## Quick Start

```bash
# Full run — all assays, parquet output
python toxcast_binary_from_ctx_api.py

# Test run — first 50 assays only, CSV output
python toxcast_binary_from_ctx_api.py --max_assays 50 --save_csv

# Resume interrupted run (skips already-downloaded AEIDs)
python toxcast_binary_from_ctx_api.py

# Force re-download everything
python toxcast_binary_from_ctx_api.py --overwrite
```
---

## Output Files

All files are saved under `--out_dir` (default: `toxcast_out/`).

| File | Description |
|------|-------------|
| `aeid_to_assay_name.csv` | AEID → assay name mapping table |
| `dtxsid_smiles.parquet` / `.csv` | DTXSID → SMILES mapping table |
| `by_aeid/aeid_{n}.parquet` | Per-AEID intermediate files (used for resume) |
| `toxcast_binary_wide.parquet` / `.csv` | **Final output**: rows = molecules (SMILES), columns = assay names, values = 0 / 1 / NaN |
| `failed_aeids.txt` | AEIDs that failed to download (if any) |
| `empty_aeids.txt` | AEIDs with no data returned by the API (if any) |

---

## Notes

- **`hitc` cleaning**: values in `(-1e-6, 0)` or `(1, 1+1e-6)` are treated as floating-point noise and snapped to `0` or `1` respectively.
- **Duplicate (DTXSID, AEID) records**: aggregated by taking the `max` of `hitc` (most conservative / active-biased).
- **Missing SMILES**: compounds for which no SMILES could be retrieved are excluded from the final wide table.
