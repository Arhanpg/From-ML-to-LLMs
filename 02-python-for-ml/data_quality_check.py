"""Inspect a small DataFrame for common data-quality issues.

Run from the repository root:
    python 02-python-for-ml/data_quality_check.py

The example uses synthetic data and does not download anything.
"""

import pandas as pd


def summarize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row of quality statistics per column."""
    if frame.columns.has_duplicates:
        raise ValueError("Column names must be unique for an unambiguous report")

    rows = []
    for column in frame.columns:
        values = frame[column]
        non_missing = values.dropna()
        rows.append(
            {
                "column": column,
                "dtype": str(values.dtype),
                "row_count": len(values),
                "missing_count": int(values.isna().sum()),
                "missing_percent": (
                    float(values.isna().mean() * 100.0) if len(values) else 0.0
                ),
                "unique_non_missing": int(non_missing.nunique()),
                "duplicate_rows_in_column": int(non_missing.duplicated().sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    frame = pd.DataFrame(
        {
            "age": [21, 22, None, 22, 35, 21],
            "city": ["Dharwad", "Hubballi", "Dharwad", "Hubballi", None, "Dharwad"],
            "score": [82.0, 91.0, 77.0, 91.0, 68.0, 82.0],
        }
    )

    print("Sample data:")
    print(frame)
    print("\nQuality summary:")
    print(summarize_dataframe(frame).to_string(index=False))

    duplicate_records = frame.duplicated(keep=False)
    print("\nRows participating in exact duplicate records:")
    print(frame.loc[duplicate_records])

    print(
        "\nReminder: duplicates are not automatically errors. Decide what counts "
        "as a duplicate based on the meaning of each row."
    )


if __name__ == "__main__":
    main()
