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
    return _dedupe_and_sort(results)

  aL, bL = reduce_linear(lhs) # Reduce the left-hand side to get 'a' and 'b'
  aR, bR = reduce_linear(rhs) # Reduce the right-hand side to get 'a' and 'b'
  a = aL - aR # Calculate the coefficient of 'x'
  b = bL - bR # Calculate the constant term

  if a != 0: # If the coefficient of 'x' is not zero
    return _dedupe_and_sort([utils.fix_zero(-b / a)]) # Return the single root

  if b == 0: # If both 'a' and 'b' are zero (e.g., 0=0)
    return ["ALL_REAL_NUMBERS"] # The equation is an identity

  return [] # If 'a' is zero and 'b' is not zero (e.g., 0=5), there is no solution

def solve_inequality(equation):
  """
  Solves a linear inequality and returns solution intervals.
  """
  lhs, op, rhs = parsing.split_inequality(equation)
  lhs_ast = parsing.parse_expr(lhs)
  rhs_ast = parsing.parse_expr(rhs)
  expr_ast = parsing.ast.BinOp(
    left=cast(parsing.ast.expr, lhs_ast),
    op=parsing.ast.Sub(),
    right=cast(parsing.ast.expr, rhs_ast),
  )

  cases = parsing.build_abs_cases(expr_ast)
  if cases:
    intervals = []
    for case_expr, constraints in cases:
      a, b = parsing.linearize_ast(case_expr)
      intervals.extend(_solve_linear_inequality_from_coeffs(a, b, op))
      intervals = _filter_intervals_by_constraints(intervals, constraints)
    return _merge_intervals(intervals)

  a, b = parsing.linearize_ast(expr_ast)
  return _solve_linear_inequality_from_coeffs(a, b, op)

def format_intervals(intervals):
  if not intervals:
    return "no solution"
  if intervals == [(-float("inf"), float("inf"), False, False)]:
    return "all real numbers"

  parts = []
  for low, high, inc_low, inc_high in intervals:
    left = "[" if inc_low else "("
    right = "]" if inc_high else ")"
    low_text = "-∞" if low == -float("inf") else _fmt_number(low)
    high_text = "∞" if high == float("inf") else _fmt_number(high)
    if low == high:
      parts.append(f"{{{low_text}}}")
    else:
      parts.append(f"{left}{low_text}, {high_text}{right}")
  return " U ".join(parts)

def format_solutions(solutions):
  if solutions == ["ALL_REAL_NUMBERS"]:
    return "all real numbers"
  if not solutions:
    return "no solution"
  if len(solutions) == 1:
    return f"{{{_fmt_number(solutions[0])}}}"
  return "{" + ", ".join(_fmt_number(v) for v in solutions) + "}"

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

def _solve_linear_inequality_from_coeffs(a, b, op):
  """
  Solves a*x + b op 0 and returns a list of intervals.
  """
  if a == 0:
    truth = _compare_constant(b, op)
    return [(-float("inf"), float("inf"), False, False)] if truth else []

  cutoff = -b / a
  if op == "<":
    return [(-float("inf"), cutoff, False, False)] if a > 0 else [(cutoff, float("inf"), False, False)]
  if op == "<=":
    return [(-float("inf"), cutoff, False, True)] if a > 0 else [(cutoff, float("inf"), True, False)]
  if op == ">":
    return [(cutoff, float("inf"), False, False)] if a > 0 else [(-float("inf"), cutoff, False, False)]
  if op == ">=":
    return [(cutoff, float("inf"), True, False)] if a > 0 else [(-float("inf"), cutoff, False, True)]
  raise ValueError("Unsupported inequality operator.")

def _compare_constant(value, op):
  if op == "<":
    return value < 0
  if op == "<=":
    return value <= 0
  if op == ">":
    return value > 0
  if op == ">=":
    return value >= 0
  raise ValueError("Unsupported inequality operator.")

def _filter_intervals_by_constraints(intervals, constraints):
  if not constraints:
    return intervals
  filtered = intervals
  for constraint in constraints:
    constraint_intervals = _constraint_to_intervals(constraint)
    filtered = _intersect_interval_lists(filtered, constraint_intervals)
    if not filtered:
      break
  return filtered

def _pick_sample(low, high):
  if low == -float("inf") and high == float("inf"):
    return 0.0
  if low == -float("inf"):
    return high - 1.0
  if high == float("inf"):
    return low + 1.0
  return (low + high) / 2.0

def _constraint_to_intervals(constraint):
  if not isinstance(constraint, parsing.ast.Compare):
    raise ValueError("Expected Compare node.")
  if len(constraint.ops) != 1 or len(constraint.comparators) != 1:
    raise ValueError("Only single comparisons supported.")

  diff = parsing.ast.BinOp(
    left=constraint.left, op=parsing.ast.Sub(), right=constraint.comparators[0]
  )
  a, b = parsing.linearize_ast(diff)
  op = constraint.ops[0]
  if isinstance(op, parsing.ast.Lt):
    return _solve_linear_inequality_from_coeffs(a, b, "<")
  if isinstance(op, parsing.ast.LtE):
    return _solve_linear_inequality_from_coeffs(a, b, "<=")
  if isinstance(op, parsing.ast.Gt):
    return _solve_linear_inequality_from_coeffs(a, b, ">")
  if isinstance(op, parsing.ast.GtE):
    return _solve_linear_inequality_from_coeffs(a, b, ">=")
  if isinstance(op, parsing.ast.Eq):
    roots = solve_linear_from_coeffs(a, b)
    if roots == ["ALL_REAL_NUMBERS"]:
      return [(-float("inf"), float("inf"), False, False)]
    return [(r, r, True, True) for r in roots if isinstance(r, (int, float))]
  raise ValueError("Unsupported comparison.")

def _intersect_interval_lists(a_list, b_list):
  result = []
  for a in a_list:
    for b in b_list:
      hit = _intersect_intervals(a, b)
      if hit is not None:
        result.append(hit)
  return _merge_intervals(result)

def _intersect_intervals(a, b):
  a_low, a_high, a_inc_low, a_inc_high = a
  b_low, b_high, b_inc_low, b_inc_high = b

  low = max(a_low, b_low)
  high = min(a_high, b_high)
  if low > high:
    return None
  if low == high:
    inc = _is_inclusive_at(low, a_low, a_inc_low, a_high, a_inc_high) and _is_inclusive_at(
      low, b_low, b_inc_low, b_high, b_inc_high
    )
    return (low, high, inc, inc) if inc else None

  inc_low = _is_inclusive_at(low, a_low, a_inc_low, a_high, a_inc_high) and _is_inclusive_at(
    low, b_low, b_inc_low, b_high, b_inc_high
  )
  inc_high = _is_inclusive_at(
    high, a_low, a_inc_low, a_high, a_inc_high
  ) and _is_inclusive_at(high, b_low, b_inc_low, b_high, b_inc_high)
  return (low, high, inc_low, inc_high)

def _is_inclusive_at(value, low, inc_low, high, inc_high):
  if value == low:
    return inc_low
  if value == high:
    return inc_high
  return True

def _merge_intervals(intervals, eps=1e-9):
  if not intervals:
    return []
  intervals.sort(key=lambda x: x[0])
  merged = [intervals[0]]
  for low, high, inc_low, inc_high in intervals[1:]:
    last_low, last_high, last_inc_low, last_inc_high = merged[-1]
    if low <= last_high + eps:
      merged[-1] = (
        last_low,
        max(last_high, high),
        last_inc_low,
        last_inc_high or inc_high,
      )
    else:
      merged.append((low, high, inc_low, inc_high))
  return merged

def _fmt_number(value):
  if abs(value) < 1e-9:
    return "0"
  if float(value).is_integer():
    return str(int(value))
  return str(value)

def _dedupe_and_sort(results, eps=1e-9):
  if not results:
    return []

  if "ALL_REAL_NUMBERS" in results:
    return ["ALL_REAL_NUMBERS"]

  numeric = [r for r in results if isinstance(r, (int, float))]
  if not numeric:
    return []

  numeric.sort()
  deduped = [numeric[0]]
  for value in numeric[1:]:
    if abs(value - deduped[-1]) > eps:
      deduped.append(value)

  return [utils.fix_zero(v) for v in deduped]

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
