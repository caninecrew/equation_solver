"""
Dispatcher module.

Routes an equation string to the correct solver based on structure checks
(for now, presence of abs()), independent of any left/right orientation.
"""

from absolute_solve import solve_absolute
from linear_solve import solve_linear

def solve(equation: str) -> list:
    """
    Main entry point for equation solving.

    Args:
        equation: Raw equation string (e.g., "2x+3=7").

    Returns:
        A list of solutions, or a special token like "ALL_REAL_NUMBERS".
    """
    if "abs(" in equation:
        # Route to the absolute value solver if abs(...) is present.
        return solve_absolute(equation)

    # Default to the linear solver for non-absolute equations.
    return solve_linear(equation)
