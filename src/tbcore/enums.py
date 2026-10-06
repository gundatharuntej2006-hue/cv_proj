"""Enumerations for classification, triage, orientations, quality statuses, and governance."""

from enum import Enum


class CXRClass(str, Enum):
    HEALTHY = "HEALTHY"
    SICK_NON_TB = "SICK_NON_TB"
    TB = "TB"


class TriageCategory(str, Enum):
    TB = "TB"
    REFER = "REFER"
    NOT_TB = "NOT_TB"


class QualityStatus(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"


class ViewOrientation(str, Enum):
    PA = "PA"
    AP = "AP"
    LATERAL = "LATERAL"
    UNKNOWN = "UNKNOWN"


class CaseStatus(str, Enum):
    ANALYSED = "ANALYSED"
    REJECTED_GATE = "REJECTED_GATE"
    UNREADABLE = "UNREADABLE"


class SplitName(str, Enum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    INTERNAL_TEST = "internal_test"
    CHALLENGE_TEST = "challenge_test"
    EXTERNAL_TEST = "external_test"
    EXCLUDED_DUPLICATE = "excluded_duplicate"


class OperatingPoint(str, Enum):
    LOCKED_DEFAULT = "locked_default"
    USER_OVERRIDE = "user_override"


class CIMethod(str, Enum):
    WILSON = "wilson"
    BOOTSTRAP_PERCENTILE = "bootstrap_percentile"
