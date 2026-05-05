from dataclasses import dataclass


@dataclass
class GestureResult:
    label: str
    confidence: float
