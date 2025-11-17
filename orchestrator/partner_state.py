from enum import Enum

class PartnerState(Enum):
    IDLE = "idle"
    AWAITING_PROBLEM = "awaiting_problem"        # ← Добавлено
    DECONSTRUCTING = "deconstructing"
    REFRAMING = "reframing"
    STRATEGIZING = "strategizing"
