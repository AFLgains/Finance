from dataclasses import dataclass
from typing import List

@dataclass
class performance_outcome:
    pct_change: List[float]
    strategy_name: str
