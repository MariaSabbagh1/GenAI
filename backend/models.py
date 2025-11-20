from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    device_id: str = Field(..., description="Identifier for the device/manual")
    question: str = Field(..., description="User's question about the device")


class Source(BaseModel):
    page: Optional[int] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
