"""
Absolute value solving module.

Handles equations with abs(A) = B and abs(A) = abs(C) where A and C are linear.
"""

from validation import get_sides
from linear_reduce import reduce_linear
from linear_solve import solve_linear_from_coeffs

def is_abs_wrapped(expr: str) -> bool:
    """
    Check whether an expression is wrapped as abs(...).
    """
    return expr.startswith("abs(") and expr.endswith(")")

def unwrap_abs(expr: str) -> str:
    """
    Remove the surrounding abs( and ) from an expression.
    """
    return expr[4:-1]

def solve_absolute(equation: str) -> list:
    """
    Solve absolute value equations using linear reductions.

    Args:
        equation: Equation string like "abs(x-3)=5" or "abs(x-3)=abs(2x+1)".

    Returns:
        Sorted list of unique solutions.
    """
    # Split and validate the equation into two sides.
    side_a, side_b = get_sides(equation)

    side_a_is_abs = is_abs_wrapped(side_a)
    side_b_is_abs = is_abs_wrapped(side_b)

    solutions = []

    if side_a_is_abs and side_b_is_abs:
        # Case 1: abs(A) = abs(C) -> A = C or A = -C.
        A = unwrap_abs(side_a)
        C = unwrap_abs(side_b)

        a1, b1 = reduce_linear(A)
        a2, b2 = reduce_linear(C)

        # A = C.
        sol1 = solve_linear_from_coeffs(a1 - a2, b1 - b2)
        solutions.extend(sol1)

        # A = -C.
        sol2 = solve_linear_from_coeffs(a1 + a2, b1 + b2)
        solutions.extend(sol2)

        return sorted(set(solutions))

    if side_a_is_abs:
        # Case 2: abs(A) = B.
        A = unwrap_abs(side_a)
        other = side_b
    elif side_b_is_abs:
        # Case 2: B = abs(A).
        A = unwrap_abs(side_b)
        other = side_a
    else:
        raise ValueError("No absolute value detected")

    # Reduce both sides to coefficients.
    aA, bA = reduce_linear(A)
    aB, bB = reduce_linear(other)

    if aB != 0:
        raise ValueError("Right-hand side of abs equation must be a constant")

    B = bB
    if B < 0:
        # abs(...) cannot equal a negative value.
        return []

    # Case A = B.
    sol1 = solve_linear_from_coeffs(aA, bA - B)
    solutions.extend(sol1)

    # Case A = -B.
    sol2 = solve_linear_from_coeffs(aA, bA + B)
    solutions.extend(sol2)

    return sorted(set(solutions))
