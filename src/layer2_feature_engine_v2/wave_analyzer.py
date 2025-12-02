from collections import deque
import logging

logger = logging.getLogger(__name__)

class WaveAnalyzer:
    """
    Analyzes Wave Strength (Impulse vs Pullback) using rolling buffers.
    """
    def __init__(self, window=10):
        self.window = window
        max_buf = max(window, 20) # Ensure enough for cum_delta_20
        
        # Rolling buffers
        self.delta_buffer = deque(maxlen=max_buf)
        self.tick_speed_buffer = deque(maxlen=max_buf)
        self.close_buffer = deque(maxlen=max_buf)
        self.fvg_creation_buffer = deque(maxlen=max_buf) 
        self.bos_creation_buffer = deque(maxlen=max_buf) 
        
        # Normalization buffers (Window 50)
        self.max_disp_buffer = deque(maxlen=50)
        self.max_delta_buffer = deque(maxlen=50)
        self.max_speed_buffer = deque(maxlen=50)
        
        self.last_fvg_count = 0
        
    def update(self, bar, smc_state, current_fvg_count: int) -> dict:
        """
        Update with new bar and SMC state.
        """
        # Update rolling buffers
        self.delta_buffer.append(bar.delta)
        self.tick_speed_buffer.append(bar.tick_speed)
        self.close_buffer.append(bar.c)
        
        # FVG Creation
        new_fvg_created = 1 if current_fvg_count > self.last_fvg_count else 0
        self.last_fvg_count = current_fvg_count
        self.fvg_creation_buffer.append(new_fvg_created)
        
        # BOS Creation
        is_bos = 1 if (smc_state.internal_bos_bull or smc_state.internal_bos_bear) else 0
        self.bos_creation_buffer.append(is_bos)
        
        # Helper for rolling sum/avg
        def get_rolling_sum(buffer, n):
            return sum(list(buffer)[-n:]) if len(buffer) >= n else sum(buffer)
            
        def get_rolling_avg(buffer, n):
            return (sum(list(buffer)[-n:]) / n) if len(buffer) >= n else (sum(buffer) / len(buffer) if buffer else 0)

        # 1. Cumulative Deltas
        cum_delta_5 = get_rolling_sum(self.delta_buffer, 5)
        cum_delta_10 = get_rolling_sum(self.delta_buffer, 10)
        cum_delta_20 = get_rolling_sum(self.delta_buffer, 20)
        
        # 2. Impulse Strength Inputs (10 bars)
        price_disp = abs(bar.c - self.close_buffer[0]) if len(self.close_buffer) >= self.window else 0.0
        
        avg_tick_speed_10 = get_rolling_avg(self.tick_speed_buffer, 10)
        count_fvg_10 = get_rolling_sum(self.fvg_creation_buffer, 10)
        count_bos_10 = get_rolling_sum(self.bos_creation_buffer, 10)
        
        # Normalization Helper (Rolling Max 50)
        self.max_disp_buffer.append(price_disp)
        self.max_delta_buffer.append(abs(cum_delta_10))
        self.max_speed_buffer.append(avg_tick_speed_10)
        
        def normalize(val, buffer):
            max_val = max(buffer) if buffer else 1.0
            if max_val < 1e-6: return 0.0
            return min(val / max_val, 1.0) # Clamp 0-1
            
        norm_disp = normalize(price_disp, self.max_disp_buffer)
        norm_delta = normalize(abs(cum_delta_10), self.max_delta_buffer)
        norm_speed = normalize(avg_tick_speed_10, self.max_speed_buffer)
        
        # Impulse Formula
        impulse_raw = (
            0.35 * norm_disp +
            0.25 * norm_delta +
            0.20 * norm_speed +
            0.10 * (min(count_fvg_10, 2) / 2) +
            0.10 * (min(count_bos_10, 2) / 2)
        )
        impulse_strength = max(0.0, min(impulse_raw * 100, 100.0))
        
        # 3. Pullback Strength Inputs (10 bars)
        pb_raw = (
            0.25 * (1.0 - norm_delta) +
            0.25 * (1.0 - norm_speed) +
            0.25 * (1.0 - min(count_fvg_10, 1)) + # Penalty if FVG exists
            0.25 * (1.0 - min(count_bos_10, 1))   # Penalty if BOS exists
        )
        pullback_strength = max(0.0, min(pb_raw * 100, 100.0))
        
        return {
            'impulse_strength': impulse_strength,
            'pullback_strength': pullback_strength,
            'cum_delta_5': cum_delta_5,
            'cum_delta_10': cum_delta_10,
            'cum_delta_20': cum_delta_20
        }
