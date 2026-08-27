from typing import Literal, Tuple


class ExecutionMode:
    ISOLATED = "isolated"
    THREADED = "threaded"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.ISOLATED, cls.THREADED


ExecutionModeValue = Literal["isolated", "threaded"]


class AdmissionStrategy:
    RAMP = "ramp"
    IMMEDIATE = "immediate"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.RAMP, cls.IMMEDIATE


AdmissionStrategyValue = Literal["ramp", "immediate"]


DEFAULT_EXECUTION_MODE = ExecutionMode.ISOLATED
DEFAULT_ADMISSION_STRATEGY = AdmissionStrategy.RAMP
DEFAULT_MAX_CONCURRENT_DOCUMENTS = 50
MAX_CONCURRENT_DOCUMENTS = 150
DEFAULT_SETTLEMENT_GRACE_SECONDS = 30.0
MAX_SETTLEMENT_GRACE_SECONDS = 60.0
DEFAULT_MAX_INFLIGHT_LIABILITY_CENTS = 8
RAMP_STAGES = (32, 96, 150)
