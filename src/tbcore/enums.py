# Enumerations for classification, triage, orientations, and quality statuses.

from enum import Enum


class CXRClass(str, Enum):
    # 3-Class diagnostic label for chest radiographs.
    HEALTHY = "HEALTHY"
    SICK_NON_TB = "SICK_NON_TB"
    TB = "TB"


class TriageCategory(str, Enum):
    # Conformal triage category for clinical workflow routing.
    TB = "TB"
    REFER = "REFER"
    NOT_TB = "NOT_TB"


class QualityStatus(str, Enum):
    # Input radiograph quality gate status.
    PASS = "PASS"
    REJECT = "REJECT"


class ViewOrientation(str, Enum):
    # Radiograph projection orientation.
    PA = "PA"
    AP = "AP"
    LATERAL = "LATERAL"
    UNKNOWN = "UNKNOWN"
