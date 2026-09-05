from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ML_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ML_DIR / "models"

RECOVERY_MODEL_FILE = (
    MODEL_DIR / "recovery_model.joblib"
)

RISK_MODEL_FILE = (
    MODEL_DIR / "risk_model.joblib"
)


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("Loading RecoverZ ML models...")

recovery_model = joblib.load(
    RECOVERY_MODEL_FILE
)

risk_model = joblib.load(
    RISK_MODEL_FILE
)

print("Recovery model loaded.")

print("Risk model loaded.")


# ============================================================
# RECOVERY MODEL FEATURE CONTRACT
# ============================================================

RECOVERY_NUMERIC_FEATURES = [
    "total_amount",
    "promo_amount",
    "shipment_fee",
    "net_amount",
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_month",
    "is_weekend",
    "previous_transaction_count",
    "previous_success_count",
    "previous_failure_count",
    "previous_success_rate",
    "days_since_previous_transaction",
    "session_event_count",
    "unique_event_types",
    "search_count",
    "booking_count",
    "promo_page_count",
    "add_promo_count",
    "session_duration_seconds",
    "purchase_intent_score",
    "year",
]


RECOVERY_CATEGORICAL_FEATURES = [
    "payment_method",
    "gender",
    "device_type",
    "device_version",
    "home_country",
    "traffic_source",
    "product_gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
]


RECOVERY_FEATURES = (
    RECOVERY_NUMERIC_FEATURES
    + RECOVERY_CATEGORICAL_FEATURES
)


# ============================================================
# RISK MODEL FEATURE CONTRACT
# ============================================================

RISK_NUMERIC_FEATURES = [
    "account_age_days",
    "avg_monthly_spend",
    "merchant_risk_score",
    "transaction_amount",
    "ip_risk_score",
    "txn_count_1h",
    "txn_count_24h",
    "failed_txn_count_24h",
    "geo_distance_from_last_txn",
    "amount_deviation_from_user_mean",
    "is_international",
]


RISK_CATEGORICAL_FEATURES = [
    "credit_score_band",
    "kyc_level",
    "payment_channel",
    "device_type",
]


RISK_FEATURES = (
    RISK_NUMERIC_FEATURES
    + RISK_CATEGORICAL_FEATURES
)


# ============================================================
# HELPERS
# ============================================================

def _numeric(
    value,
    default=0.0
):
    """
    Convert a value to float.

    If unavailable or invalid, use the supplied
    conservative/default value.
    """

    try:

        if value is None:
            return float(default)

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return float(default)


def _categorical(
    value,
    default="unknown"
):
    """
    Convert categorical values to strings.
    """

    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


# ============================================================
# RECOVERY FEATURE ADAPTER
# ============================================================

def build_recovery_features(
    case
):
    """
    Convert a RecoverZ payment case into the feature
    contract used by the real-data recovery model.

    `case` can be a dictionary or an object with attributes.
    """

    def get(
        key,
        default=None
    ):

        if isinstance(case, dict):

            return case.get(
                key,
                default
            )

        return getattr(
            case,
            key,
            default
        )

    amount = _numeric(
        get(
            "amount",
            get(
                "total_amount",
                0
            )
        )
    )

    promo_amount = _numeric(
        get(
            "promo_amount",
            0
        )
    )

    shipment_fee = _numeric(
        get(
            "shipment_fee",
            0
        )
    )

    net_amount = _numeric(
        get(
            "net_amount",
            amount
        )
    )

    previous_success_rate = _numeric(
        get(
            "previous_success_rate",
            0.5
        ),
        default=0.5
    )

    previous_transaction_count = _numeric(
        get(
            "previous_transaction_count",
            0
        )
    )

    previous_success_count = _numeric(
        get(
            "previous_success_count",
            round(
                previous_transaction_count
                * previous_success_rate
            )
        )
    )

    previous_failure_count = _numeric(
        get(
            "previous_failure_count",
            max(
                0,
                previous_transaction_count
                - previous_success_count
            )
        )
    )

    features = {

        "total_amount":
            amount,

        "promo_amount":
            promo_amount,

        "shipment_fee":
            shipment_fee,

        "net_amount":
            net_amount,

        "transaction_hour":
            _numeric(
                get(
                    "transaction_hour",
                    12
                )
            ),

        "transaction_day_of_week":
            _numeric(
                get(
                    "transaction_day_of_week",
                    0
                )
            ),

        "transaction_month":
            _numeric(
                get(
                    "transaction_month",
                    1
                )
            ),

        "is_weekend":
            _numeric(
                get(
                    "is_weekend",
                    0
                )
            ),

        "previous_transaction_count":
            previous_transaction_count,

        "previous_success_count":
            previous_success_count,

        "previous_failure_count":
            previous_failure_count,

        "previous_success_rate":
            previous_success_rate,

        "days_since_previous_transaction":
            _numeric(
                get(
                    "days_since_previous_transaction",
                    30
                )
            ),

        "session_event_count":
            _numeric(
                get(
                    "session_event_count",
                    0
                )
            ),

        "unique_event_types":
            _numeric(
                get(
                    "unique_event_types",
                    0
                )
            ),

        "search_count":
            _numeric(
                get(
                    "search_count",
                    0
                )
            ),

        "booking_count":
            _numeric(
                get(
                    "booking_count",
                    0
                )
            ),

        "promo_page_count":
            _numeric(
                get(
                    "promo_page_count",
                    0
                )
            ),

        "add_promo_count":
            _numeric(
                get(
                    "add_promo_count",
                    0
                )
            ),

        "session_duration_seconds":
            _numeric(
                get(
                    "session_duration_seconds",
                    0
                )
            ),

        "purchase_intent_score":
            _numeric(
                get(
                    "purchase_intent_score",
                    0
                )
            ),

        "year":
            _numeric(
                get(
                    "year",
                    2026
                )
            ),

        "payment_method":
            _categorical(
                get(
                    "payment_method",
                    "unknown"
                )
            ),

        "gender":
            _categorical(
                get(
                    "gender",
                    "unknown"
                )
            ),

        "device_type":
            _categorical(
                get(
                    "device_type",
                    "unknown"
                )
            ),

        "device_version":
            _categorical(
                get(
                    "device_version",
                    "unknown"
                )
            ),

        "home_country":
            _categorical(
                get(
                    "home_country",
                    "unknown"
                )
            ),

        "traffic_source":
            _categorical(
                get(
                    "traffic_source",
                    "unknown"
                )
            ),

        "product_gender":
            _categorical(
                get(
                    "product_gender",
                    "unknown"
                )
            ),

        "masterCategory":
            _categorical(
                get(
                    "masterCategory",
                    "unknown"
                )
            ),

        "subCategory":
            _categorical(
                get(
                    "subCategory",
                    "unknown"
                )
            ),

        "articleType":
            _categorical(
                get(
                    "articleType",
                    "unknown"
                )
            ),

        "baseColour":
            _categorical(
                get(
                    "baseColour",
                    "unknown"
                )
            ),

        "season":
            _categorical(
                get(
                    "season",
                    "unknown"
                )
            ),

        "usage":
            _categorical(
                get(
                    "usage",
                    "unknown"
                )
            ),
    }

    return pd.DataFrame(
        [
            features
        ],
        columns=RECOVERY_FEATURES
    )


# ============================================================
# RISK FEATURE ADAPTER
# ============================================================

def build_risk_features(
    case
):
    """
    Convert RecoverZ case/risk context into the
    payment-risk model feature contract.
    """

    def get(
        key,
        default=None
    ):

        if isinstance(case, dict):

            return case.get(
                key,
                default
            )

        return getattr(
            case,
            key,
            default
        )

    amount = _numeric(
        get(
            "amount",
            get(
                "transaction_amount",
                0
            )
        )
    )

    features = {

        "account_age_days":
            _numeric(
                get(
                    "account_age_days",
                    365
                )
            ),

        "avg_monthly_spend":
            _numeric(
                get(
                    "avg_monthly_spend",
                    amount
                )
            ),

        "merchant_risk_score":
            _numeric(
                get(
                    "merchant_risk_score",
                    0.10
                )
            ),

        "transaction_amount":
            amount,

        "ip_risk_score":
            _numeric(
                get(
                    "ip_risk_score",
                    0.10
                )
            ),

        "txn_count_1h":
            _numeric(
                get(
                    "txn_count_1h",
                    1
                )
            ),

        "txn_count_24h":
            _numeric(
                get(
                    "txn_count_24h",
                    1
                )
            ),

        "failed_txn_count_24h":
            _numeric(
                get(
                    "failed_txn_count_24h",
                    0
                )
            ),

        "geo_distance_from_last_txn":
            _numeric(
                get(
                    "geo_distance_from_last_txn",
                    0
                )
            ),

        "amount_deviation_from_user_mean":
            _numeric(
                get(
                    "amount_deviation_from_user_mean",
                    0
                )
            ),

        "is_international":
            _numeric(
                get(
                    "is_international",
                    0
                )
            ),

        "credit_score_band":
            _categorical(
                get(
                    "credit_score_band",
                    3
                )
            ),

        "kyc_level":
            _categorical(
                get(
                    "kyc_level",
                    2
                )
            ),

        "payment_channel":
            _categorical(
                get(
                    "payment_channel",
                    get(
                        "payment_method",
                        "unknown"
                    )
                )
            ),

        "device_type":
            _categorical(
                get(
                    "device_type",
                    "unknown"
                )
            ),
    }

    return pd.DataFrame(
        [
            features
        ],
        columns=RISK_FEATURES
    )


# ============================================================
# PREDICTION API
# ============================================================

def predict_recovery(
    case
):
    """
    Return calibrated recovery probability.
    """

    X = build_recovery_features(
        case
    )

    probability = float(
        recovery_model.predict_proba(X)[0][1]
    )

    return probability


def predict_risk(
    case
):
    """
    Return calibrated fraud/risk probability.
    """

    X = build_risk_features(
        case
    )

    probability = float(
        risk_model.predict_proba(X)[0][1]
    )

    return probability


# ============================================================
# COMBINED DECISION SIGNALS
# ============================================================

def score_case(
    case
):
    """
    Generate both ML signals and expected recovery value.

    Policy decisions are intentionally NOT made here.
    """

    recovery_probability = (
        predict_recovery(case)
    )

    risk_probability = (
        predict_risk(case)
    )

    if isinstance(case, dict):

        amount = case.get(
            "amount",
            case.get(
                "total_amount",
                0
            )
        )

    else:

        amount = getattr(
            case,
            "amount",
            getattr(
                case,
                "total_amount",
                0
            )
        )

    amount = _numeric(
        amount
    )

    expected_recovery_value = (
        amount
        * recovery_probability
    )

    return {

        "recovery_probability":
            round(
                recovery_probability,
                4
            ),

        "risk_probability":
            round(
                risk_probability,
                4
            ),

        "expected_recovery_value":
            round(
                expected_recovery_value,
                2
            ),
    }