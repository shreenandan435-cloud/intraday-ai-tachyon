import logging
from enum import Enum, auto

class SystemState(Enum):
    INIT = auto()
    PRE_MARKET = auto()
    AUTHENTICATING = auto()
    CONNECTING = auto()
    SEEDING = auto()
    SCANNING = auto()
    POSITION_OPEN = auto()
    RISK_LOCK = auto()
    DATA_DEGRADED = auto()
    BROKER_ERROR = auto()
    SHUTTING_DOWN = auto()
    CLOSED = auto()

class TachyonStateMachine:
    def __init__(self):
        self.current_state = SystemState.INIT
        self.logger = logging.getLogger("Tachyon.State")

    def transition_to(self, new_state: SystemState, reason: str = ""):
        """Safely transitions the engine to a new state and logs the change."""
        valid_transition = True
        
        # Example safety lock: Cannot go from CLOSED back to SCANNING
        if self.current_state == SystemState.CLOSED and new_state != SystemState.INIT:
            valid_transition = False
            
        if not valid_transition:
            self.logger.error(f"[!] Invalid state transition attempted: {self.current_state.name} -> {new_state.name}")
            return False

        old_state = self.current_state
        self.current_state = new_state
        reason_text = f" ({reason})" if reason else ""
        self.logger.info(f"State Transition: {old_state.name} -> {new_state.name}{reason_text}")
        return True

    def get_state(self):
        return self.current_state
