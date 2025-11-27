"""
Volume Profile Configuration
Supports daily or session-based profiles (Asia/Europe/NY)
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class SessionDef:
    """Session definition with start/end hours"""
    name: str
    start_hour: int  # 0-23
    end_hour: int    # 0-24 (exclusive)


@dataclass
class VPConfig:
    """
    Volume Profile Configuration
    
    Supports:
    - Daily profile (1 profile per day)
    - Session profile (multiple profiles per day: Asia/Europe/NY)
    """
    tick_size: float
    mode: Literal["daily", "session"] = "daily"
    session_mode: Literal["asia_europe_ny", "custom"] = "asia_europe_ny"
    sessions: List[SessionDef] = None
    
    # Value area percentage (default 70%)
    value_area_pct: float = 0.70
    
    def __post_init__(self):
        """Set default sessions if not provided"""
        if self.sessions is None and self.mode == "session":
            if self.session_mode == "asia_europe_ny":
                self.sessions = [
                    SessionDef("Asia", start_hour=0, end_hour=8),
                    SessionDef("Europe", start_hour=8, end_hour=16),
                    SessionDef("NY", start_hour=16, end_hour=24),
                ]
            else:
                # Fallback to full day
                self.sessions = [SessionDef("Day", start_hour=0, end_hour=24)]


# Default configs for common instruments
GC_M1_VP_CONFIG = VPConfig(
    tick_size=0.1,
    mode="session",
    session_mode="asia_europe_ny",
    sessions=[
        SessionDef("Asia", start_hour=0, end_hour=8),
        SessionDef("Europe", start_hour=8, end_hour=16),
        SessionDef("NY", start_hour=16, end_hour=24),
    ]
)

NQ_M1_VP_CONFIG = VPConfig(
    tick_size=0.25,
    mode="session",
    session_mode="asia_europe_ny",
    sessions=[
        SessionDef("Asia", start_hour=0, end_hour=8),
        SessionDef("Europe", start_hour=8, end_hour=16),
        SessionDef("NY", start_hour=16, end_hour=24),
    ]
)

# Daily mode config (simpler, 1 profile per day)
GC_M1_VP_DAILY = VPConfig(
    tick_size=0.1,
    mode="daily"
)
