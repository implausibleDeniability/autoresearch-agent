from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Mapping, Optional, Tuple

from src.cost_metering.accounting import CostReport, MeteringOutcome
from src.evaluation.models import PIIItem
from src.evaluation.results import EvaluationTrace


class DocumentStatus:
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.COMPLETED, cls.FAILED, cls.NOT_ATTEMPTED


DocumentStatusValue = Literal["completed", "failed", "not_attempted"]


class ResultStatus:
    COMPLETE = "complete"
    PARTIAL = "partial"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.COMPLETE, cls.PARTIAL


ResultStatusValue = Literal["complete", "partial"]


class LifecycleStatus:
    RUNNING = "running"
    TERMINAL = "terminal"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.RUNNING, cls.TERMINAL


LifecycleStatusValue = Literal["running", "terminal"]


@dataclass(frozen=True)
class DocumentExecution:
    ordinal: int
    document_id: str
    status: DocumentStatusValue
    source_tokens: int
    usage: Optional[CostReport] = None
    usage_complete: bool = False
    latency_seconds: Optional[float] = None
    predictions: Tuple[PIIItem, ...] = ()
    failure_category: str = ""
    error_message: str = ""
    retryable: bool = False

    @property
    def usage_status(self) -> str:
        return "complete" if self.usage_complete else "incomplete"


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    dataset: str
    texts: Mapping[str, str]
    source_tokens: int
    documents: Tuple[DocumentExecution, ...]
    trace: EvaluationTrace
    cost: MeteringOutcome
    lifecycle_status: LifecycleStatusValue
    termination_category: str
    started_at: str
    updated_at: str

    def __post_init__(self) -> None:
        document_ids = tuple(document.document_id for document in self.documents)
        if document_ids != tuple(self.texts):
            raise ValueError("document ledger must contain every dataset document exactly once in order")
        if any(document.status not in DocumentStatus.all() for document in self.documents):
            raise ValueError("document ledger contains an unsupported status")
        if any(
            document.predictions and document.status != DocumentStatus.COMPLETED
            for document in self.documents
        ):
            raise ValueError("only completed documents may retain predictions")
        if self.lifecycle_status == LifecycleStatus.TERMINAL:
            complete = all(document.status == DocumentStatus.COMPLETED for document in self.documents)
            if (complete and self.cost.status == "complete") != (self.result_status == ResultStatus.COMPLETE):
                raise ValueError("terminal result completeness is inconsistent")

    @property
    def result_status(self) -> Optional[ResultStatusValue]:
        if self.lifecycle_status == LifecycleStatus.RUNNING:
            return None
        complete = all(document.status == DocumentStatus.COMPLETED for document in self.documents)
        return ResultStatus.COMPLETE if complete and self.cost.status == "complete" else ResultStatus.PARTIAL

    @property
    def completed_source_tokens(self) -> int:
        return sum(
            document.source_tokens
            for document in self.documents
            if document.status == DocumentStatus.COMPLETED
        )

    @classmethod
    def timestamp(cls) -> str:
        return datetime.now(timezone.utc).isoformat()
