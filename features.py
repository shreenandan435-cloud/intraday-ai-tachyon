import numpy as np
from numba import njit

@njit(cache=True)
def compute_microprice_velocity(b_px, b_vol, a_px, a_vol, prev_microprice):
    if b_vol[0] + a_vol[0] == 0:
        return prev_microprice, 0.0
        
    microprice = (b_px[0] * a_vol[0] + a_px[0] * b_vol[0]) / (b_vol[0] + a_vol[0])
    velocity_bps = ((microprice - prev_microprice) / prev_microprice) * 10000.0 if prev_microprice > 0 else 0.0
    
    return microprice, velocity_bps

@njit(cache=True)
def compute_multi_level_ofi(prev_b_px, prev_b_vol, prev_a_px, prev_a_vol, curr_b_px, curr_b_vol, curr_a_px, curr_a_vol):
    ofi = 0.0
    for i in range(5):
        b_inc = (curr_b_px[i] >= prev_b_px[i]) * curr_b_vol[i]
        b_dec = (curr_b_px[i] <= prev_b_px[i]) * prev_b_vol[i]
        
        a_inc = (curr_a_px[i] <= prev_a_px[i]) * curr_a_vol[i]
        a_dec = (curr_a_px[i] >= prev_a_px[i]) * prev_a_vol[i]
        
        ofi += (b_inc - b_dec) - (a_inc - a_dec)
    return ofi
