from pathlib import Path
import ast
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
ECOMMERCE_DIR = RAW_DIR / "ecommerce"

TRANSACTIONS_FILE = ECOMMERCE_DIR / "transactions.csv"
CUSTOMER_FILE = ECOMMERCE_DIR / "customer.csv"
CLICKSTREAM_FILE = ECOMMERCE_DIR / "click_stream.csv"
PRODUCT_FILE = ECOMMERCE_DIR / "product.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "recovery_training.csv"


# ============================================================
# 1. LOAD TRANSACTIONS
# ============================================================

def load_transactions():

    print("\n[1/7] Loading transactions...")

    columns = [
        "created_at",
        "customer_id",
        "booking_id",
        "session_id",
        "product_metadata",
        "payment_method",
        "payment_status",
        "promo_amount",
        "promo_code",
        "shipment_fee",
        "shipment_date_limit",
        "shipment_location_lat",
        "shipment_location_long",
        "total_amount",
    ]

    df = pd.read_csv(
        TRANSACTIONS_FILE,
        usecols=columns
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    print(f"Transactions: {len(df):,}")

    return df


# ============================================================
# 2. TRANSACTION FEATURES
# ============================================================

def build_transaction_features(df):

    print("\n[2/7] Building transaction features...")

    df = df.copy()

    # Monetary features
    df["total_amount"] = pd.to_numeric(
        df["total_amount"],
        errors="coerce"
    ).fillna(0)

    df["promo_amount"] = pd.to_numeric(
        df["promo_amount"],
        errors="coerce"
    ).fillna(0)

    df["shipment_fee"] = pd.to_numeric(
        df["shipment_fee"],
        errors="coerce"
    ).fillna(0)

    df["net_amount"] = (
        df["total_amount"]
        - df["promo_amount"]
        + df["shipment_fee"]
    )

    # Time features
    df["transaction_hour"] = (
        df["created_at"].dt.hour
    )

    df["transaction_day_of_week"] = (
        df["created_at"].dt.dayofweek
    )

    df["transaction_month"] = (
        df["created_at"].dt.month
    )

    df["is_weekend"] = (
        df["transaction_day_of_week"] >= 5
    ).astype(int)

    # Payment outcome
    df["payment_success"] = (
        df["payment_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "success"
    ).astype(int)

    return df


# ============================================================
# 3. CUSTOMER HISTORY
# ============================================================

def build_customer_history(df):

    print("\n[3/7] Building leakage-safe customer history...")

    df = df.sort_values(
        ["customer_id", "created_at"]
    ).copy()

    grouped = df.groupby("customer_id")

    # Number of transactions BEFORE current transaction
    df["previous_transaction_count"] = (
        grouped.cumcount()
    )

    # Cumulative successes INCLUDING current transaction
    cumulative_success = (
        grouped["payment_success"]
        .cumsum()
    )

    # Remove current transaction's result.
    # This guarantees no target leakage.
    df["previous_success_count"] = (
        cumulative_success
        - df["payment_success"]
    )

    df["previous_failure_count"] = (
        df["previous_transaction_count"]
        - df["previous_success_count"]
    )

    df["previous_success_rate"] = np.where(
        df["previous_transaction_count"] > 0,
        df["previous_success_count"]
        / df["previous_transaction_count"],
        0.0
    )

    df["previous_transaction_at"] = (
        grouped["created_at"]
        .shift(1)
    )

    df["days_since_previous_transaction"] = (
        df["created_at"]
        - df["previous_transaction_at"]
    ).dt.total_seconds() / 86400

    df["days_since_previous_transaction"] = (
        df["days_since_previous_transaction"]
        .fillna(999)
        .clip(lower=0)
    )

    return df

# ============================================================
# 4. CUSTOMER PROFILE
# ============================================================

def load_customer_features():

    print("\n[4/7] Loading customer profiles...")

    columns = [
        "customer_id",
        "gender",
        "device_type",
        "device_version",
        "home_country",
        "first_join_date",
    ]

    customer = pd.read_csv(
        CUSTOMER_FILE,
        usecols=columns
    )

    customer["first_join_date"] = pd.to_datetime(
        customer["first_join_date"],
        errors="coerce"
    )

    return customer


# ============================================================
# 5. CLICKSTREAM AGGREGATION
# ============================================================

def build_clickstream_features():

    print("\n[5/7] Processing 12.8M clickstream events...")

    results = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            CLICKSTREAM_FILE,
            chunksize=500_000
        ),
        start=1
    ):

        print(
            f"  Processing clickstream chunk "
            f"{chunk_number}..."
        )

        chunk["event_name"] = (
            chunk["event_name"]
            .astype(str)
            .str.upper()
        )

        chunk["event_time"] = pd.to_datetime(
            chunk["event_time"],
            errors="coerce"
        )

        # Event indicators
        chunk["is_search"] = (
            chunk["event_name"] == "SEARCH"
        ).astype("int8")

        chunk["is_booking"] = (
            chunk["event_name"] == "BOOKING"
        ).astype("int8")

        chunk["is_promo_page"] = (
            chunk["event_name"] == "PROMO_PAGE"
        ).astype("int8")

        chunk["is_add_promo"] = (
            chunk["event_name"] == "ADD_PROMO"
        ).astype("int8")

        grouped = chunk.groupby(
            "session_id"
        ).agg(
            session_event_count=(
                "event_id",
                "count"
            ),

            unique_event_types=(
                "event_name",
                "nunique"
            ),

            session_start=(
                "event_time",
                "min"
            ),

            session_end=(
                "event_time",
                "max"
            ),

            search_count=(
                "is_search",
                "sum"
            ),

            booking_count=(
                "is_booking",
                "sum"
            ),

            promo_page_count=(
                "is_promo_page",
                "sum"
            ),

            add_promo_count=(
                "is_add_promo",
                "sum"
            ),

            traffic_source=(
                "traffic_source",
                "first"
            ),
        ).reset_index()

        results.append(grouped)

        print(
            f"  Finished chunk {chunk_number}"
        )

    features = pd.concat(
        results,
        ignore_index=True
    )

    # Combine duplicate session records created
    # by chunk processing.
    features = (
        features
        .groupby("session_id")
        .agg(
            session_event_count=(
                "session_event_count",
                "sum"
            ),

            unique_event_types=(
                "unique_event_types",
                "max"
            ),

            session_start=(
                "session_start",
                "min"
            ),

            session_end=(
                "session_end",
                "max"
            ),

            search_count=(
                "search_count",
                "sum"
            ),

            booking_count=(
                "booking_count",
                "sum"
            ),

            promo_page_count=(
                "promo_page_count",
                "sum"
            ),

            add_promo_count=(
                "add_promo_count",
                "sum"
            ),

            traffic_source=(
                "traffic_source",
                "first"
            ),
        )
        .reset_index()
    )

    features["session_duration_seconds"] = (
        features["session_end"]
        - features["session_start"]
    ).dt.total_seconds()

    features["session_duration_seconds"] = (
        features["session_duration_seconds"]
        .fillna(0)
        .clip(lower=0)
    )

    features["purchase_intent_score"] = (
        features["search_count"] * 0.10
        + features["booking_count"] * 0.50
        + features["promo_page_count"] * 0.15
        + features["add_promo_count"] * 0.25
    )

    return features


# ============================================================
# 6. PRODUCT FEATURES
# ============================================================

def build_product_features(df):

    print("\n[6/7] Extracting product information...")

    def extract_product_id(value):

        try:

            data = ast.literal_eval(str(value))

            if isinstance(data, list) and len(data) > 0:
                return data[0].get("product_id")

            if isinstance(data, dict):
                return data.get("product_id")

        except Exception:
            pass

        return np.nan

    df["product_id"] = (
        df["product_metadata"]
        .apply(extract_product_id)
    )

    product = pd.read_csv(
        PRODUCT_FILE,
        usecols=[
            "id",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "year",
            "usage",
        ]
    )

    product = product.rename(
        columns={
            "id": "product_id",
            "gender": "product_gender",
        }
    )

    df = df.merge(
        product,
        on="product_id",
        how="left"
    )

    return df


# ============================================================
# 7. RECOVERY PROXY
# ============================================================

def create_recovery_proxy(df):

    print("\n[7/7] Creating recovery-proxy target...")

    # Sort by customer and time.
    df = df.sort_values(
        ["customer_id", "created_at"]
    ).copy()

    # --------------------------------------------------------
    # Find the NEXT successful transaction for each customer.
    #
    # IMPORTANT:
    # bfill() looks forward within each customer's timeline.
    # --------------------------------------------------------

    next_success_time = (
        df["created_at"]
        .where(df["payment_success"] == 1)
        .groupby(df["customer_id"])
        .bfill()
    )

    # Time from current transaction to next success.
    hours_to_next_success = (
        next_success_time - df["created_at"]
    ).dt.total_seconds() / 3600

    # --------------------------------------------------------
    # Recovery proxy:
    #
    # Failed payment
    #      +
    # next successful payment
    #      +
    # within 14 days
    # --------------------------------------------------------

    df["recovery_proxy"] = np.nan

    failed_mask = (
        df["payment_success"] == 0
    )

    recovered_mask = (
        failed_mask
        & hours_to_next_success.notna()
        & (hours_to_next_success > 0)
        & (hours_to_next_success <= 14 * 24)
    )

    not_recovered_mask = (
        failed_mask
        & ~recovered_mask
    )

    df.loc[
        recovered_mask,
        "recovery_proxy"
    ] = 1.0

    df.loc[
        not_recovered_mask,
        "recovery_proxy"
    ] = 0.0

    # Successful transactions are not recovery cases.
    df.loc[
        df["payment_success"] == 1,
        "recovery_proxy"
    ] = np.nan

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    failed_count = int(
        failed_mask.sum()
    )

    recovered_count = int(
        recovered_mask.sum()
    )

    not_recovered_count = int(
        not_recovered_mask.sum()
    )

    print(
        f"Failed transactions: "
        f"{failed_count:,}"
    )

    print(
        f"Failed transactions followed by "
        f"success within 14 days: "
        f"{recovered_count:,}"
    )

    print(
        f"Failed transactions without "
        f"14-day recovery: "
        f"{not_recovered_count:,}"
    )

    print(
        f"Recovery rate among failures: "
        f"{recovered_count / failed_count:.2%}"
    )

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load
    df = load_transactions()

    # Basic transaction features
    df = build_transaction_features(df)

    # Historical features
    df = build_customer_history(df)

    # Customer profile
    customer = load_customer_features()

    df = df.merge(
        customer,
        on="customer_id",
        how="left"
    )

    # Time-safe clickstream features
    # These were generated separately using DuckDB and only
    # include events available at or before transaction time.

    TIMESAFE_CLICKSTREAM_FILE = (
        PROCESSED_DIR
        / "session_features_timesafe.parquet"
    )

    print(
        "\nLoading time-safe clickstream features..."
    )

    clickstream = pd.read_parquet(
        TIMESAFE_CLICKSTREAM_FILE
    )

# Normalize timestamps before joining.
#
# Raw transactions use UTC timestamps with timezone information,
# while the time-safe parquet was written as timezone-naive timestamps.
# Convert both to the same UTC-naive representation.

    df["transaction_time"] = (
    pd.to_datetime(
        df["created_at"],
        utc=True
    )
    .dt.tz_localize(None)
)

    clickstream["transaction_time"] = (
    pd.to_datetime(
        clickstream["transaction_time"],
        utc=True
    )
    .dt.tz_localize(None)
)

# Join using customer + session + normalized transaction timestamp.
    df = df.merge(
    clickstream,
    on=[
        "customer_id",
        "session_id",
        "transaction_time",
    ],
    how="left",
)

    df = df.drop(
    columns=["transaction_time"],
    errors="ignore"
)

    df = df.drop(
        columns=["transaction_time"],
        errors="ignore"
    )

    # Product information
    df = build_product_features(df)

    # Recovery proxy
    df = create_recovery_proxy(df)

    # Fill clickstream values
    clickstream_columns = [
        "session_event_count",
        "unique_event_types",
        "search_count",
        "booking_count",
        "promo_page_count",
        "add_promo_count",
        "session_duration_seconds",
        "purchase_intent_score",
    ]

    for column in clickstream_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna(0)
            )

    # Remove PII / raw fields
    drop_columns = [
        "first_name",
        "last_name",
        "username",
        "email",
        "home_location_lat",
        "home_location_long",
        "shipment_location_lat",
        "shipment_location_long",
        "product_metadata",
        "previous_transaction_at",
        "session_start",
        "session_end",
    ]

    df = df.drop(
        columns=[
            column
            for column in drop_columns
            if column in df.columns
        ]
    )

    # Chronological ordering
    df = df.sort_values(
        "created_at"
    )

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("RECOVERY DATASET CREATED")
    print("=" * 70)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    print("\nRecovery proxy distribution:")

    print(
        df["recovery_proxy"]
        .value_counts(dropna=False)
    )

    print("\nFeature columns:")

    print(
        df.columns.tolist()
    )


if __name__ == "__main__":
    main()