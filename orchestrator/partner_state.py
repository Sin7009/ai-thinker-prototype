from enum import Enum

class PartnerState(Enum):
    IDLE = "idle"
    AWAITING_PROBLEM = "awaiting_problem"
    DECONSTRUCTION = "deconstruction"
    HYPOTHESIS_FIELD = "hypothesis_field"
    STRESS_TESTING = "stress_testing"
    SYNTHESIS = "synthesis"
    ASSIMILATION = "assimilation"
