"""
Audit script for analyzing the search and feedback events database.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from backend.taxonomy_data import lexical_category_probabilities


def main() -> None:
    """
    Main function to audit the events database.
    """
    db_path = Path("backend/runtime/events.db")
    if not db_path.exists():
        print("Database not found:", db_path)
        return

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    counts = {
        "search_events": cur.execute("SELECT COUNT(*) FROM search_events").fetchone()[0],
        "feedback_events": cur.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0],
        "category_totals": cur.execute("SELECT COUNT(*) FROM category_totals").fetchone()[0],
        "tag_totals": cur.execute("SELECT COUNT(*) FROM tag_totals").fetchone()[0],
    }

    rows = cur.execute(
        """
        SELECT
          se.id,
          se.query,
          se.predicted_category,
          se.created_at,
          fe.true_category AS confirmed_category,
          fe.confidence AS confirmed_confidence,
          fe.created_at AS confirmed_at
        FROM search_events se
        LEFT JOIN feedback_events fe
          ON fe.id = (
            SELECT id
            FROM feedback_events
            WHERE query = se.query
            ORDER BY id DESC
            LIMIT 1
          )
        ORDER BY se.id ASC
        """
    ).fetchall()

    pred_vs_confirm_total = 0
    pred_vs_confirm_match = 0
    unconfirmed = 0
    pred_vs_lex_match = 0
    confirm_vs_lex_match = 0

    flagged_unconfirmed = []
    flagged_confirmed = []

    for row in rows:
        query = row["query"]
        predicted = row["predicted_category"]
        confirmed = row["confirmed_category"]

        lexical_probs = lexical_category_probabilities(query)
        lexical_top = max(lexical_probs.items(), key=lambda item: item[1])[0]
        lexical_score = lexical_probs[lexical_top]

        if predicted == lexical_top:
            pred_vs_lex_match += 1

        if confirmed is None:
            unconfirmed += 1
            if predicted != lexical_top and lexical_score >= 0.14:
                flagged_unconfirmed.append(
                    {
                        "id": row["id"],
                        "query": query,
                        "predicted": predicted,
                        "lexical_top": lexical_top,
                        "lexical_score": round(lexical_score, 3),
                    }
                )
        else:
            pred_vs_confirm_total += 1
            if predicted == confirmed:
                pred_vs_confirm_match += 1
            if confirmed == lexical_top:
                confirm_vs_lex_match += 1
            elif lexical_score >= 0.14:
                flagged_confirmed.append(
                    {
                        "id": row["id"],
                        "query": query,
                        "predicted": predicted,
                        "confirmed": confirmed,
                        "lexical_top": lexical_top,
                        "lexical_score": round(lexical_score, 3),
                    }
                )

    print("=== DATABASE SUMMARY ===")
    print("path:", db_path)
    for key, value in counts.items():
        print(f"{key}: {value}")

    print("\n=== QUALITY SUMMARY ===")
    if pred_vs_confirm_total:
        print(
            "predicted == confirmed:",
            f"{pred_vs_confirm_match}/{pred_vs_confirm_total}",
            f"({pred_vs_confirm_match / pred_vs_confirm_total:.1%})",
        )
        print(
            "confirmed == lexical_top:",
            f"{confirm_vs_lex_match}/{pred_vs_confirm_total}",
            f"({confirm_vs_lex_match / pred_vs_confirm_total:.1%})",
        )
    else:
        print("predicted == confirmed: n/a (no confirmed rows)")

    print(
        "predicted == lexical_top:",
        f"{pred_vs_lex_match}/{len(rows)}",
        f"({(pred_vs_lex_match / len(rows)) if rows else 0:.1%})",
    )
    print("unconfirmed rows:", unconfirmed)

    print("\n=== FLAGGED UNCONFIRMED (needs review) ===")
    if not flagged_unconfirmed:
        print("none")
    else:
        for item in flagged_unconfirmed:
            print(item)

    print("\n=== FLAGGED CONFIRMED (human vs lexical disagreement) ===")
    if not flagged_confirmed:
        print("none")
    else:
        for item in flagged_confirmed:
            print(item)

    print("\n=== PREDICTED CATEGORY DISTRIBUTION ===")
    pred_counts = Counter(row["predicted_category"] for row in rows)
    for category, count in pred_counts.most_common():
        print(category, count)

    print("\n=== CONFIRMED CATEGORY DISTRIBUTION ===")
    confirmed_counts = Counter(row["confirmed_category"] for row in rows if row["confirmed_category"])
    for category, count in confirmed_counts.most_common():
        print(category, count)


if __name__ == "__main__":
    main()
