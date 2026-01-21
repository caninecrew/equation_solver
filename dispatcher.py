from absolute_solve import solve_absolute
from linear_solve import solve_linear

def solve(equation: str) -> list:
    if "abs(" in equation:
        return solve_absolute(equation)
    return solve_linear(equation)
