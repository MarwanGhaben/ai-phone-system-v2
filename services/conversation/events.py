"""Minimal speech-event contracts shared by STT adapters and dialogue consumers."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


@dataclass(frozen=True)
class UtteranceIdentity:
    """Locally issued identity for one committed transcript.

    ``connection_epoch`` is opaque and unique to one STT instance/connection.
    ``commit_sequence`` increases for every canonical commit in that epoch.
    Provider text is deliberately not part of the identity.
    """

    provider: str
    connection_epoch: str
    commit_sequence: int

    def __post_init__(self) -> None:
        if not self.provider or not self.connection_epoch:
            raise ValueError("utterance identity scope must be non-empty")
        if (
            isinstance(self.commit_sequence, bool)
            or not isinstance(self.commit_sequence, int)
            or self.commit_sequence < 1
        ):
            raise ValueError("commit sequence must be a positive integer")


class MetadataDisposition(str, Enum):
    """Safe classification for non-actionable provider metadata."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    REPEATED = "repeated"
    MALFORMED = "malformed"
    OLD_EPOCH = "old_epoch"
    UNSUPPORTED = "unsupported"
    EVICTED = "evicted"


@dataclass(frozen=True)
class WordTiming:
    """Validated timing only; recognized word text is intentionally excluded."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class UtteranceMetadataEvent:
    """Bounded, non-actionable enrichment or diagnostic evidence.

    Metadata arriving after a transcript was handed to dialogue is retained here;
    it never mutates or replays that transcript.
    """

    connection_epoch: str
    disposition: MetadataDisposition
    provider_event_type: str
    provenance: str
    utterance_id: Optional[UtteranceIdentity] = None
    language: Optional[str] = None
    language_confidence: Optional[float] = None
    confidence: Optional[float] = None
    word_timings: Tuple[WordTiming, ...] = ()
    invalid_fields: Tuple[str, ...] = ()
