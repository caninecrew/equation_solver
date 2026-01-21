from solve import solve

tests = [
    ("2x + 3 = 7", [2.0]),
    ("2x - 3 = 7", [5.0]),
    ("x + x = 10", [5.0]),
    ("2*(x+1) = 8", [3.0]),
    ("(2x + 7) / 8 = 1", [0.5]),
    ("2x = 10", [5.0]),
    ("3(x+1) = 12", [3.0]),
    ("abs(-7) = 7", ["ALL_REAL_NUMBERS"]),
    ("x + abs(-3) = 0", [-3.0]),
    ("x/2 = 3", [6.0]),
    ("0 = 0", ["ALL_REAL_NUMBERS"]),
    ("0 = 5", []),
]

for expr, expected in tests:
    try:
        result = solve(expr)
    except Exception as exc:
        result = f"ERROR: {type(exc).__name__}: {exc}"
    print(f"{expr} -> {result} (expected {expected})")