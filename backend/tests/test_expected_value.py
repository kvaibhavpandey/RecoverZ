def test_expected_value():
    amount = 2499
    probability = 0.91
    assert round(amount * probability, 2) == 2274.09
