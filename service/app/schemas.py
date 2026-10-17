"""Pydantic-схемы ответа API. Поля verdict/violations/type/timecode_sec/description/evidence_frames —
строго по разделу 10 ТЗ; rule/confidence/type_code — наши расширения (ПЛАН_ForteBank.md, 5.6)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str = Field(description="JPEG в base64 без префикса data:")


class Rule(BaseModel):
    document: str
    clause: str
    url: str
    quote: Optional[str] = None
    note: Optional[str] = None


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)
    # расширения
    type_code: Optional[str] = None
    confidence: Optional[float] = None
    rule: Optional[Rule] = None
    end_sec: Optional[float] = None


class JobCreated(BaseModel):
    job_id: str
    status: Literal["processing"] = "processing"


class Progress(BaseModel):
    stage: str
    percent: Optional[int] = None


class JobResponse(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Optional[Literal["no_violation", "violation"]] = None
    violations: Optional[list[Violation]] = None
    error: Optional[str] = None
    progress: Optional[Progress] = None
    meta: Optional[dict] = None
