import utils
import parsing
from typing import cast

def reduce_linear(expr):
    """
    Reduces a linear expression to its coefficients (a, b) for the form a*x + b.

    Args:
        expr (str): The linear expression as a string.

    Returns:
        tuple[float, float]: A tuple (a, b) representing the coefficients of the linear expression a*x + b.
    """
    return parsing.reduce_linear(expr)

def solve_linear(equation):
  """
  Solves a linear equation of the form ax + b = 0.
  """

  lhs, rhs = parsing.split_equation(equation) # Split the equation into left and right sides
  lhs_ast = parsing.parse_expr(lhs)
  rhs_ast = parsing.parse_expr(rhs)
  expr_ast = parsing.ast.BinOp(
    left=cast(parsing.ast.expr, lhs_ast),
    op=parsing.ast.Sub(),
    right=cast(parsing.ast.expr, rhs_ast),
  )
  cases = parsing.build_abs_cases(expr_ast)
  if cases:
    results = []
    for case_expr, constraints in cases:
      a, b = parsing.linearize_ast(case_expr)
      solutions = solve_linear_from_coeffs(a, b)
      for sol in solutions:
        if sol == "ALL_REAL_NUMBERS":
          if all(_constraint_always_true(c) for c in constraints):
            results.append(sol)
          continue
        if all(parsing.eval_constraint(c, sol) for c in constraints):
          results.append(utils.fix_zero(sol))
    return results

  aL, bL = reduce_linear(lhs) # Reduce the left-hand side to get 'a' and 'b'
  aR, bR = reduce_linear(rhs) # Reduce the right-hand side to get 'a' and 'b'
  a = aL - aR # Calculate the coefficient of 'x'
  b = bL - bR # Calculate the constant term

  if a != 0: # If the coefficient of 'x' is not zero
    return [utils.fix_zero(-b / a)] # Return the single root

  if b == 0: # If both 'a' and 'b' are zero (e.g., 0=0)
    return ["ALL_REAL_NUMBERS"] # The equation is an identity

  return [] # If 'a' is zero and 'b' is not zero (e.g., 0=5), there is no solution

def _constraint_always_true(constraint):
  if not isinstance(constraint, parsing.ast.Compare):
    return False
  if len(constraint.ops) != 1 or len(constraint.comparators) != 1:
    return False

  diff = parsing.ast.BinOp(
    left=constraint.left, op=parsing.ast.Sub(), right=constraint.comparators[0]
  )
  a, b = parsing.linearize_ast(diff)
  if a != 0.0:
    return False

  op = constraint.ops[0]
  if isinstance(op, parsing.ast.GtE):
    return b >= 0.0
  if isinstance(op, parsing.ast.Gt):
    return b > 0.0
  if isinstance(op, parsing.ast.LtE):
    return b <= 0.0
  if isinstance(op, parsing.ast.Lt):
    return b < 0.0
  if isinstance(op, parsing.ast.Eq):
    return b == 0.0
  return False

def solve_linear_from_coeffs(a, b):
    """
    Solves a*x + b = 0
    """
    # Handles the three linear cases: single root, identity, or no solution.
    if a != 0:
        return [-b / a]
    if b == 0:
        return ["ALL_REAL_NUMBERS"]
    return []
