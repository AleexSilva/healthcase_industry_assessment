from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
import calendar
from typing import Dict, Any, List, Optional, Tuple
import yaml
import os
import pandas as pd


# ---------------------------
# Configuration
# ---------------------------

RENAME_MAP = {
    "Revenue": "Revenue",
    "ID": "UserID",
    "AdvancedMD Appointment UID": "AppointmentID",
    "1/2": "Datasource",
    "Service Date": "ServiceDate",
}

FINAL_COL_ORDER = ["Revenue", "UserID", "AppointmentID", "Datasource", "ServiceDate"]

ALLOWED_DATASOURCES = {1, 2}


# ---------------------------
# Data Quality Report Structures
# ---------------------------

@dataclass
class DQMetric:
    name: str
    value: Any
    severity: str = "INFO"  # INFO | WARN | ERROR
    details: Optional[Dict[str, Any]] = None


@dataclass
class DQReport:
    generated_at_utc: str
    input_path: str
    row_count_raw: int
    row_count_clean: int
    metrics: List[DQMetric]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "input_path": self.input_path,
            "row_count_raw": self.row_count_raw,
            "row_count_clean": self.row_count_clean,
            "metrics": [asdict(m) for m in self.metrics],
        }


# ---------------------------
# Helpers
# ---------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def last_day_of_month(year: int, month: int) -> date:
    last_dom = calendar.monthrange(year, month)[1]
    return date(year, month, last_dom)


def safe_to_int_nullable(s: pd.Series) -> pd.Series:
    # Use pandas nullable integer type to keep NaNs
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def safe_to_float_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Float64")


def normalize_service_date(dt: pd.Series) -> pd.Series:
    # Coerce to datetime; keep NaT if invalid; normalize to midnight for consistency
    out = pd.to_datetime(dt, errors="coerce")
    return out.dt.normalize()

def load_yaml_config(path: str = "config/config.yml") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("pipeline", cfg)


# ---------------------------
# Core Pipeline Steps
# ---------------------------

def load_excel(input_path: str, sheet_name: Optional[str | int] = 0) -> pd.DataFrame:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    if isinstance(df, dict):
        first_key = next(iter(df))
        logging.warning("read_excel returned multiple sheets; using first sheet: '%s'", first_key)
        df = df[first_key]
    return df


def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in RENAME_MAP.keys() if c not in df.columns]
    return (len(missing) == 0, missing)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column headers (common Excel issue)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Rename
    df = df.rename(columns=RENAME_MAP)

    # Keep only the columns we care about (in case extra columns exist)
    keep = [RENAME_MAP[c] for c in RENAME_MAP.keys() if RENAME_MAP[c] in df.columns]
    df = df[keep]

    # Reorder exactly as requested
    df = df.reindex(columns=FINAL_COL_ORDER)

    return df

def clean_revenue_text(s: pd.Series) -> pd.Series:
    """
    Cleans Revenue values that come in as strings with stray characters like ´ or `.
    Also removes common formatting noise (commas, spaces, currency symbols).
    Keeps negatives if present.
    """
    if s is None:
        return s

    # Work in string space safely
    x = s.astype("string")

    # Remove the specific marks you mentioned + common variants
    x = x.str.replace("´", "", regex=False)
    x = x.str.replace("`", "", regex=False)
    x = x.str.replace("’", "", regex=False)   # curly apostrophe
    x = x.str.replace("'", "", regex=False)   # straight apostrophe

    # Remove common formatting noise
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace("$", "", regex=False)
    x = x.str.replace(" ", "", regex=False)

    # If there are any other non-numeric characters, drop them (except . and -)
    x = x.str.replace(r"[^0-9\.\-]", "", regex=True)

    return x


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Revenue: numeric float (nullable)
    df["Revenue"] = safe_to_float_nullable(clean_revenue_text(df["Revenue"]))

    # IDs: nullable ints
    df["UserID"] = safe_to_int_nullable(df["UserID"])
    df["AppointmentID"] = safe_to_int_nullable(df["AppointmentID"])

    # Datasource: nullable int
    df["Datasource"] = safe_to_int_nullable(df["Datasource"])

    # ServiceDate: datetime normalized
    df["ServiceDate"] = normalize_service_date(df["ServiceDate"])

    return df


def drop_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df2 = df.drop_duplicates()
    return df2, before - len(df2)


def dq_checks(df_raw: pd.DataFrame, df: pd.DataFrame, input_path: str) -> DQReport:
    metrics: List[DQMetric] = []

    # Basic null checks
    for col in FINAL_COL_ORDER:
        nulls = int(df[col].isna().sum())
        sev = "WARN" if nulls > 0 else "INFO"
        metrics.append(DQMetric(name=f"null_count.{col}", value=nulls, severity=sev))

    # Revenue checks
    non_numeric_rev = int(df["Revenue"].isna().sum())
    if non_numeric_rev > 0:
        metrics.append(
            DQMetric(
                name="revenue.non_numeric_or_missing",
                value=non_numeric_rev,
                severity="WARN",
                details={"note": "Revenue coercion produced nulls; check source formatting."},
            )
        )

    rev_le_zero = int((df["Revenue"].fillna(0) <= 0).sum())
    # Note: this counts nulls as <=0 after fillna(0). If you want strictly <=0 ignoring nulls, adjust.
    if rev_le_zero > 0:
        metrics.append(
            DQMetric(
                name="revenue.le_zero_count",
                value=rev_le_zero,
                severity="WARN",
                details={"rule": "Revenue should be > 0 for billable encounters."},
            )
        )

    # Datasource validity
    invalid_ds = df.loc[df["Datasource"].notna() & ~df["Datasource"].isin(list(ALLOWED_DATASOURCES)), "Datasource"]
    invalid_ds_count = int(invalid_ds.shape[0])
    if invalid_ds_count > 0:
        metrics.append(
            DQMetric(
                name="datasource.invalid_value_count",
                value=invalid_ds_count,
                severity="ERROR",
                details={"allowed": sorted(list(ALLOWED_DATASOURCES)), "examples": invalid_ds.head(10).tolist()},
            )
        )

    # Duplicate AppointmentID check (common source consistency issue)
    appt_nonnull = df[df["AppointmentID"].notna()]
    dup_appt_rows = int(appt_nonnull.duplicated(subset=["AppointmentID"], keep=False).sum())
    if dup_appt_rows > 0:
        metrics.append(
            DQMetric(
                name="appointment_id.duplicate_rows",
                value=dup_appt_rows,
                severity="WARN",
                details={"note": "Same AppointmentID appears multiple times. Could be legit (line items) or duplication."},
            )
        )

    # Date sanity
    min_dt = df["ServiceDate"].min()
    max_dt = df["ServiceDate"].max()
    metrics.append(DQMetric(name="service_date.min", value=None if pd.isna(min_dt) else str(min_dt.date())))
    metrics.append(DQMetric(name="service_date.max", value=None if pd.isna(max_dt) else str(max_dt.date())))

    # November 2025 completeness heuristics (see function below for more detail)
    nov_check = check_month_completeness(df, year=2025, month=11)
    metrics.append(
        DQMetric(
            name="month_completeness.2025-11",
            value=nov_check,
            severity="INFO" if nov_check.get("looks_complete") else "WARN",
            details={"note": "Heuristic based on observed dates & day coverage; confirm with upstream owner."},
        )
    )

    report = DQReport(
        generated_at_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        input_path=input_path,
        row_count_raw=int(len(df_raw)),
        row_count_clean=int(len(df)),
        metrics=metrics,
    )
    return report


def check_month_completeness(df: pd.DataFrame, year: int, month: int) -> Dict[str, Any]:
    """
    Heuristic completeness check:
    - Filters rows within the given month
    - Computes min/max dates observed within the month
    - Computes how many distinct calendar days in that month have >=1 record
    - Declares "looks_complete" if:
        min_date == first_day_of_month AND max_date == last_day_of_month
      (plus a soft signal if coverage is high)
    """
    if df["ServiceDate"].isna().all():
        return {"looks_complete": False, "reason": "ServiceDate all null"}

    start = pd.Timestamp(date(year, month, 1))
    end = pd.Timestamp(last_day_of_month(year, month))

    m = df[(df["ServiceDate"] >= start) & (df["ServiceDate"] <= end)].copy()
    if m.empty:
        return {"looks_complete": False, "reason": "No rows in month", "month": f"{year:04d}-{month:02d}"}

    min_d = m["ServiceDate"].min().date()
    max_d = m["ServiceDate"].max().date()

    distinct_days = int(m["ServiceDate"].dt.date.nunique())
    total_days = calendar.monthrange(year, month)[1]
    coverage_pct = float(distinct_days / total_days)

    looks_complete_strict = (min_d == date(year, month, 1)) and (max_d == last_day_of_month(year, month))
    looks_complete_soft = (max_d == last_day_of_month(year, month)) and (coverage_pct >= 0.85)

    return {
        "month": f"{year:04d}-{month:02d}",
        "rows": int(len(m)),
        "min_date": str(min_d),
        "max_date": str(max_d),
        "distinct_service_days": distinct_days,
        "days_in_month": total_days,
        "coverage_pct": round(coverage_pct, 4),
        "looks_complete": bool(looks_complete_strict or looks_complete_soft),
        "strict_complete": bool(looks_complete_strict),
        "soft_complete": bool(looks_complete_soft),
    }


def run_pipeline(
    input_path: str,
    output_path: str,
    dq_report_path: str,
    sheet_name: Optional[str] = None,
    drop_rows_with_null_critical_fields: bool = False,
) -> None:
    logging.info("Loading raw Excel: %s", input_path)
    df_raw = load_excel(input_path, sheet_name=sheet_name)

    ok, missing = validate_required_columns(df_raw)
    if not ok:
        raise ValueError(f"Missing required columns in input: {missing}")

    logging.info("Standardizing column names & order")
    df = standardize_columns(df_raw)

    logging.info("Casting datatypes")
    df = cast_types(df)

    logging.info("Dropping exact duplicate rows")
    df, dropped = drop_exact_duplicates(df)
    logging.info("Dropped %d exact duplicate rows", dropped)

    # Optional: enforce critical fields (depends on business rule)
    if drop_rows_with_null_critical_fields:
        before = len(df)
        df = df.dropna(subset=["UserID", "AppointmentID", "ServiceDate"])
        logging.info("Dropped %d rows with null critical fields", before - len(df))

    logging.info("Running data-quality checks")
    report = dq_checks(df_raw=df_raw, df=df, input_path=input_path)

    # Write cleaned output
    logging.info("Writing cleaned dataset to: %s", output_path)
    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    elif output_path.lower().endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    # Write DQ report
    logging.info("Writing data-quality report to: %s", dq_report_path)
    with open(dq_report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Log key warnings to console
    warn_metrics = [m for m in report.metrics if m.severity in ("WARN", "ERROR")]
    if warn_metrics:
        logging.warning("DQ Warnings/Errors detected (%d). Review %s.", len(warn_metrics), dq_report_path)
        for m in warn_metrics[:20]:
            logging.warning("[%s] %s = %s", m.severity, m.name, m.value)
    else:
        logging.info("No DQ warnings detected.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assessment.xlsx cleaning + DQ pipeline")
    p.add_argument("--input", required=True, help="Path to raw Assessment.xlsx")
    p.add_argument("--output", required=True, help="Path to write cleaned dataset (.csv or .parquet)")
    p.add_argument("--dq-report", required=True, help="Path to write DQ report JSON")
    p.add_argument("--sheet-name", default=None, help="Excel sheet name (optional)")
    p.add_argument("--drop-null-critical", action="store_true", help="Drop rows missing UserID/AppointmentID/ServiceDate")
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARN, ERROR")
    return p


if __name__ == "__main__":
    # Load config from YAML — no CLI args needed
    cfg = load_yaml_config("config/config.yml")

    input_path      = cfg["input"]
    output_path     = cfg["output"]
    dq_report_path  = cfg["dq_report"]
    sheet_name      = cfg.get("sheet_name", 0)
    drop_null       = bool(cfg.get("drop_null_critical", False))
    log_level       = cfg.get("log_level", "INFO")

    setup_logging(log_level)

    # Auto-create output directories if they don't exist
    for path in [output_path, dq_report_path]:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    run_pipeline(
        input_path=input_path,
        output_path=output_path,
        dq_report_path=dq_report_path,
        sheet_name=sheet_name,
        drop_rows_with_null_critical_fields=drop_null,
    )