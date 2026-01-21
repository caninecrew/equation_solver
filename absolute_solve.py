from validation import split_equation
from linear_reduce import reduce_linear
from linear_solve import solve_linear_from_coeffs

def is_abs_wrapped(expr: str) -> bool:
    return expr.startswith("abs(") and expr.endswith(")")

def unwrap_abs(expr: str) -> str:
    return expr[4:-1]

def solve_absolute(equation: str) -> list:
    lhs, rhs = split_equation(equation)

    lhs_is_abs = is_abs_wrapped(lhs)
    rhs_is_abs = is_abs_wrapped(rhs)

    solutions = []

    if lhs_is_abs and rhs_is_abs:
        A = unwrap_abs(lhs)
        C = unwrap_abs(rhs)

        a1, b1 = reduce_linear(A)
        a2, b2 = reduce_linear(C)

        sol1 = solve_linear_from_coeffs(a1 - a2, b1 - b2)
        solutions.extend(sol1)

        sol2 = solve_linear_from_coeffs(a1 + a2, b1 + b2)
        solutions.extend(sol2)

        return sorted(set(solutions))

    if lhs_is_abs:
        A = unwrap_abs(lhs)
        other = rhs
    elif rhs_is_abs:
        A = unwrap_abs(rhs)
        other = lhs
    else:
        raise ValueError("No absolute value detected")

    aA, bA = reduce_linear(A)
    aB, bB = reduce_linear(other)

    if aB != 0:
        raise ValueError("Right-hand side of abs equation must be a constant")

    B = bB
    if B < 0:
        return []

    sol1 = solve_linear_from_coeffs(aA, bA - B)
    solutions.extend(sol1)

    sol2 = solve_linear_from_coeffs(aA, bA + B)
    solutions.extend(sol2)

    return sorted(set(solutions))
