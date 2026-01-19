
from aimath.tools.calculator import Calculator

def test_fast_path():
    calc = Calculator()
    q = "square root of 123456"
    print(f"Testing: {q}")
    # Force usage of Try Simple Parse to verify regex
    parsed = calc._try_simple_parse(q)
    print(f"Parsed: {parsed}")
    
    if parsed:
        result = calc.compute(parsed)
        print(f"Computed: {result['numerical']}")

if __name__ == "__main__":
    test_fast_path()
