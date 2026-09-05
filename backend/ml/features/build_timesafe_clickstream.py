from pathlib import Path
import duckdb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
ECOMMERCE_DIR = RAW_DIR / "ecommerce"

TRANSACTIONS_FILE = ECOMMERCE_DIR / "transactions.csv"
CLICKSTREAM_FILE = ECOMMERCE_DIR / "click_stream.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "session_features_timesafe.parquet"
)


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("BUILDING TIME-SAFE CLICKSTREAM FEATURES")
    print("=" * 70)

    print(
        f"Clickstream:\n{CLICKSTREAM_FILE}"
    )

    print(
        f"Transactions:\n{TRANSACTIONS_FILE}"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    con = duckdb.connect()

    # --------------------------------------------------------
    # DuckDB configuration
    # --------------------------------------------------------

    con.execute(
        "SET memory_limit='4GB'"
    )

    con.execute(
        "SET threads=4"
    )

    # --------------------------------------------------------
    # Create time-safe session features.
    #
    # We aggregate ONLY clickstream events that happened
    # BEFORE OR AT the transaction timestamp.
    #
    # The transaction timestamp is supplied through the
    # transaction table.
    # --------------------------------------------------------

    query = f"""
    COPY (

        WITH transactions AS (

            SELECT
                customer_id,
                session_id,
                created_at::TIMESTAMPTZ AT TIME ZONE 'UTC' AS transaction_time

            FROM read_csv_auto(
                '{TRANSACTIONS_FILE.as_posix()}',
                header=true
            )

            WHERE session_id IS NOT NULL
              AND created_at IS NOT NULL

        ),

        clickstream AS (

            SELECT
                session_id,
                event_name,
                event_time::TIMESTAMPTZ AT TIME ZONE 'UTC' AS event_time,
                event_id,
                traffic_source

            FROM read_csv_auto(
                '{CLICKSTREAM_FILE.as_posix()}',
                header=true
            )

            WHERE session_id IS NOT NULL
              AND event_time IS NOT NULL

        ),

        transaction_sessions AS (

            SELECT DISTINCT
                session_id
            FROM transactions

        ),

        relevant_events AS (

            SELECT
                c.session_id,
                c.event_name,
                c.event_time,
                c.event_id,
                c.traffic_source

            FROM clickstream c

            INNER JOIN transaction_sessions ts
                ON c.session_id = ts.session_id

        ),

        session_features AS (

            SELECT

                t.customer_id,

                t.session_id,

                t.transaction_time,

                COUNT(c.event_id) AS session_event_count,

                COUNT(
                    DISTINCT c.event_name
                ) AS unique_event_types,

                SUM(
                    CASE
                        WHEN UPPER(c.event_name) = 'SEARCH'
                        THEN 1
                        ELSE 0
                    END
                ) AS search_count,

                SUM(
                    CASE
                        WHEN UPPER(c.event_name) = 'BOOKING'
                        THEN 1
                        ELSE 0
                    END
                ) AS booking_count,

                SUM(
                    CASE
                        WHEN UPPER(c.event_name) = 'PROMO_PAGE'
                        THEN 1
                        ELSE 0
                    END
                ) AS promo_page_count,

                SUM(
                    CASE
                        WHEN UPPER(c.event_name) = 'ADD_PROMO'
                        THEN 1
                        ELSE 0
                    END
                ) AS add_promo_count,

                MIN(c.event_time)
                    AS session_start_before_transaction,

                MAX(c.event_time)
                    AS last_event_before_transaction,

                MAX(c.traffic_source)
                    AS traffic_source

            FROM transactions t

            LEFT JOIN relevant_events c

                ON t.session_id = c.session_id

                AND c.event_time <= t.transaction_time

            GROUP BY

                t.customer_id,
                t.session_id,
                t.transaction_time

        )

        SELECT

            customer_id,

            session_id,

            transaction_time,

            session_event_count,

            unique_event_types,

            search_count,

            booking_count,

            promo_page_count,

            add_promo_count,

            traffic_source,

            COALESCE(
                EPOCH(
                    last_event_before_transaction
                    - session_start_before_transaction
                ),
                0
            ) AS session_duration_seconds,

            (
                search_count * 0.10
                + booking_count * 0.50
                + promo_page_count * 0.15
                + add_promo_count * 0.25
            ) AS purchase_intent_score

        FROM session_features

    )

    TO '{OUTPUT_FILE.as_posix()}'
    (FORMAT PARQUET);
    """

    print("\nRunning DuckDB query...")
    print(
        "This may take a few minutes because "
        "the clickstream contains 12.8M events."
    )

    con.execute(query)

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    result = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT session_id) AS sessions,
            AVG(session_event_count)
                AS avg_events,
            AVG(purchase_intent_score)
                AS avg_intent
        FROM read_parquet(
            '{OUTPUT_FILE.as_posix()}'
        )
        """
    ).fetchone()

    print("\n" + "=" * 70)
    print("TIME-SAFE CLICKSTREAM FEATURES CREATED")
    print("=" * 70)

    print(
        f"Rows: {result[0]:,}"
    )

    print(
        f"Unique sessions: {result[1]:,}"
    )

    print(
        f"Average events before transaction: "
        f"{result[2]:.2f}"
    )

    print(
        f"Average purchase intent: "
        f"{result[3]:.4f}"
    )

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )

    con.close()


if __name__ == "__main__":
    main()