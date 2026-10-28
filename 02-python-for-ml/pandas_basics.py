"""A small pandas workflow using in-memory data.

Run from the repository root:
    python 02-python-for-ml/pandas_basics.py
"""

import pandas as pd


def main() -> None:
    records = pd.DataFrame(
        {
            "student": ["Asha", "Ravi", "Meera", "Ravi", "Ishaan"],
            "course": ["ML", "ML", "DL", "ML", "DL"],
            "hours": [4.0, 6.0, 5.0, 6.0, None],
            "score": [78, 84, 91, 84, 73],
        }
    )

    print("First rows:")
    print(records.head())
    print("\nColumn data types:")
    print(records.dtypes)
    print("\nMissing values per column:")
    print(records.isna().sum())

    cleaned = records.copy()
    # Median filling is used only to demonstrate a method on this toy data.
    # Real projects should justify imputation using training data and context.
    median_hours = cleaned["hours"].median()
    cleaned["hours"] = cleaned["hours"].fillna(median_hours)

    high_scores = cleaned.loc[
        cleaned["score"] >= 80, ["student", "score"]
    ]
    print("\nScores of at least 80:")
    print(high_scores)

    course_summary = cleaned.groupby("course").agg(
        average_score=("score", "mean"),
        average_hours=("hours", "mean"),
        student_rows=("student", "size"),
    )
    print("\nSummary by course:")
    print(course_summary)

    duplicate_mask = cleaned.duplicated(subset=["student", "course"], keep=False)
    print("\nRows duplicated by student and course:")
    print(cleaned.loc[duplicate_mask].sort_values(["student", "course"]))


if __name__ == "__main__":
    main()
