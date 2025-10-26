from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvidenceItem(BaseModel):
    text: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0

class ExternalItem(BaseModel):
    snippet: str = ""
    url: str = ""
    source: str = ""

class EvidenceBundle(BaseModel):
    internal: List[EvidenceItem] = Field(default_factory=list)
    external: List[ExternalItem] = Field(default_factory=list)

class ClaimInput(BaseModel):
    id: Optional[int] = None
    text: str

class Verdict(BaseModel):
    label: int  # 1=Real, 0=Fake, 2=Unverifiable (if you use it)
    proba: float
    explanation: str

class AgentState(BaseModel):
    claim: ClaimInput
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    verdict: Optional[Verdict] = None
    trace: Dict[str, Any] = Field(default_factory=dict)  # timings, node outputs
