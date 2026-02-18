from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List, Literal
from datetime import datetime
from uuid import UUID


class FightEvent(BaseModel):
    """Base fight event"""
    timestamp: float  # Video timestamp in seconds
    frame_number: int
    event_type: str
    fighter_id: int  # 0 or 1


class PunchEvent(FightEvent):
    """Punch event with classification"""
    event_type: Literal["punch"] = "punch"
    punch_type: Literal["jab", "cross", "hook", "uppercut"]
    hand: Literal["left", "right"]
    outcome: Literal["landed", "missed", "blocked"]
    target: Literal["head", "body"]
    impact_score: float = Field(ge=0.0, le=1.0)
    speed: float  # m/s


class FightMetrics(BaseModel):
    """Real-time fight metrics"""
    timestamp: float
    fighter_id: int
    stance: Literal["orthodox", "southpaw", "squared"]
    balance: Literal["front_foot", "neutral", "back_foot"]
    distance: Literal["inside", "mid", "outside"]
    ring_position: tuple[float, float]  # (x, y) normalized 0-1
    guard_position: Literal["high", "low", "open"]


class RoundScore(BaseModel):
    """Round scoring with probabilities"""
    round_number: int
    fighter_0_score: int
    fighter_1_score: int
    prob_10_9_f0: float = Field(ge=0.0, le=1.0)
    prob_10_9_f1: float = Field(ge=0.0, le=1.0)
    prob_10_8_f0: float = Field(ge=0.0, le=1.0)
    prob_10_8_f1: float = Field(ge=0.0, le=1.0)
    win_prob_f0: float = Field(ge=0.0, le=1.0)
    win_prob_f1: float = Field(ge=0.0, le=1.0)


class FightCreate(BaseModel):
    """Create fight request"""
    video_url: Optional[str] = None
    youtube_url: Optional[str] = None


class FightResponse(BaseModel):
    """Fight response"""
    id: UUID
    status: str  # pending, processing, completed, completed_mock, failed, not_found
    created_at: datetime
    total_rounds: Optional[int] = None
    duration_seconds: Optional[float] = None
    video_url: Optional[str] = None
    win_probabilities: Optional[Dict[str, float]] = None
    round_scores: Optional[List[Dict[str, Any]]] = None


class FightStats(BaseModel):
    """Aggregated fight statistics"""
    fight_id: UUID
    fighter_0_punches_landed: int
    fighter_0_punches_thrown: int
    fighter_1_punches_landed: int
    fighter_1_punches_thrown: int
    fighter_0_power_punches: int
    fighter_1_power_punches: int
    rounds: List[RoundScore]
