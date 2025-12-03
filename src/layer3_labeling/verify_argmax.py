import numpy as np

def test_argmax_logic():
    # Scenario 1: Target is hit at index 2
    window_highs = np.array([100, 101, 105, 102])
    target = 104
    
    hit_idx = np.argmax(window_highs >= target)
    print(f"Scenario 1 (Hit at 2): Index={hit_idx}, Value={window_highs[hit_idx]}, IsHit={window_highs[hit_idx] >= target}")
    assert hit_idx == 2
    assert window_highs[hit_idx] >= target

    # Scenario 2: Target is NEVER hit
    window_highs = np.array([100, 101, 102, 103])
    target = 105
    
    hit_idx = np.argmax(window_highs >= target)
    print(f"Scenario 2 (Never Hit): Index={hit_idx}, Value={window_highs[hit_idx]}, IsHit={window_highs[hit_idx] >= target}")
    assert hit_idx == 0 # argmax returns 0 if all False
    assert not (window_highs[hit_idx] >= target) # Verification check must fail

    # Scenario 3: Target is hit at index 0
    window_highs = np.array([105, 101, 102, 103])
    target = 104
    
    hit_idx = np.argmax(window_highs >= target)
    print(f"Scenario 3 (Hit at 0): Index={hit_idx}, Value={window_highs[hit_idx]}, IsHit={window_highs[hit_idx] >= target}")
    assert hit_idx == 0
    assert window_highs[hit_idx] >= target

    print("\nAll scenarios passed verification logic!")

if __name__ == "__main__":
    test_argmax_logic()
