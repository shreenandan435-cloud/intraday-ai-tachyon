from orderbook_engine import OrderbookAnalyzer

def test_l2_hardening():
    analyzer = OrderbookAnalyzer()
    ltp = 100.0

    # Test 1: Empty Array
    res = analyzer.evaluate_liquidity({'best_5_buy_data': [], 'best_5_sell_data': []}, ltp)
    assert res.is_tradable == False and "Empty arrays" in res.reason, "Failed Empty Array Test"

    # Test 2: Crossed Book (Bid >= Ask)
    res = analyzer.evaluate_liquidity({
        'best_5_buy_data': [{'price': 101.0, 'quantity': 100}],
        'best_5_sell_data': [{'price': 100.0, 'quantity': 100}]
    }, ltp)
    assert res.is_tradable == False and "Crossed book" in res.reason, "Failed Crossed Book Test"

    # Test 3: Zero Quantity Trap (Prevents ZeroDivisionError)
    res = analyzer.evaluate_liquidity({
        'best_5_buy_data': [{'price': 100.0, 'quantity': 0}],
        'best_5_sell_data': [{'price': 101.0, 'quantity': 100}]
    }, ltp)
    assert res.is_tradable == False and "Zero total quantity" in res.reason, "Failed Zero Quantity Test"

    # Test 4: Missing Keys / Nulls
    res = analyzer.evaluate_liquidity({
        'best_5_buy_data': [{'price': None}], 
        'best_5_sell_data': [{'price': 101.0, 'quantity': 100}]
    }, ltp)
    assert res.is_tradable == False and "No valid populated" in res.reason, "Failed Null/Missing Key Test"

    print("ALL L2 HARDENING TESTS PASSED: Valid, Crossed, Zero-Qty, Empty, Null, Malformed.")

if __name__ == "__main__":
    test_l2_hardening()
