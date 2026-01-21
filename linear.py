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
  Solves a single-variable linear equation, including abs() cases.
  """

  lhs, rhs = parsing.split_equation(equation)
  lhs_ast = parsing.parse_expr(lhs)
  rhs_ast = parsing.parse_expr(rhs)
  expr_ast = parsing.ast.BinOp(
    left=cast(parsing.ast.expr, lhs_ast),
    op=parsing.ast.Sub(),
    right=cast(parsing.ast.expr, rhs_ast),
  )
  vars_found = parsing.get_variable_names(expr_ast)
  if len(vars_found) > 1:
    raise ValueError("Multiple variables are not supported in a single equation.")
  if len(vars_found) == 1 and "x" not in vars_found:
    var_name = next(iter(vars_found))
    expr_ast = parsing.replace_variable(expr_ast, var_name, "x")
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

  aL, bL = reduce_linear(lhs)
  aR, bR = reduce_linear(rhs)
  a = aL - aR
  b = bL - bR

  if a != 0:
    return _dedupe_and_sort([utils.fix_zero(-b / a)])

  if b == 0:
    return ["ALL_REAL_NUMBERS"]

  return []

def solve_inequality(equation):
  """
  Solves a linear inequality and returns solution intervals.
  """
  exprs, ops = parsing.split_inequality(equation)
  expr_asts = [parsing.parse_expr(expr) for expr in exprs]
  intervals = [(-float("inf"), float("inf"), False, False)]

  for idx, op in enumerate(ops):
    lhs_ast = expr_asts[idx]
    rhs_ast = expr_asts[idx + 1]
    expr_ast = parsing.ast.BinOp(
      left=cast(parsing.ast.expr, lhs_ast),
      op=parsing.ast.Sub(),
      right=cast(parsing.ast.expr, rhs_ast),
    )
    vars_found = parsing.get_variable_names(expr_ast)
    if len(vars_found) > 1:
      raise ValueError("Multiple variables are not supported in a single inequality.")
    if len(vars_found) == 1 and "x" not in vars_found:
      var_name = next(iter(vars_found))
      expr_ast = parsing.replace_variable(expr_ast, var_name, "x")

    cases = parsing.build_abs_cases(expr_ast)
    if cases:
      current = []
      for case_expr, constraints in cases:
        a, b = parsing.linearize_ast(case_expr)
        case_intervals = _solve_linear_inequality_from_coeffs(a, b, op)
        case_intervals = _filter_intervals_by_constraints(case_intervals, constraints)
        current.extend(case_intervals)
      intervals = _intersect_interval_lists(intervals, current)
      continue

    a, b = parsing.linearize_ast(expr_ast)
    current = _solve_linear_inequality_from_coeffs(a, b, op)
    intervals = _intersect_interval_lists(intervals, current)

  return _merge_intervals(intervals)

def solve_inequality_system(equation):
  """
  Returns half-space constraints for multi-variable linear inequalities.
  """
  exprs, ops = parsing.split_inequality(equation)
  expr_asts = [parsing.parse_expr(expr) for expr in exprs]
  constraints = []
  for idx, op in enumerate(ops):
    lhs_ast = expr_asts[idx]
    rhs_ast = expr_asts[idx + 1]
    cL, kL = parsing.linearize_multi_ast(lhs_ast)
    cR, kR = parsing.linearize_multi_ast(rhs_ast)
    coeffs = {}
    for key, val in cL.items():
      coeffs[key] = coeffs.get(key, 0.0) + val
    for key, val in cR.items():
      coeffs[key] = coeffs.get(key, 0.0) - val
    const = kL - kR
    constraints.append((coeffs, op, -const))
  return constraints

def format_halfspaces(constraints):
  """
  Formats constraints like "2x + y <= 3".
  """
  parts = []
  for coeffs, op, rhs in constraints:
    terms = []
    for var in sorted(coeffs.keys()):
      coef = coeffs[var]
      if abs(coef) < 1e-12:
        continue
      sign = "+" if coef > 0 else "-"
      mag = abs(coef)
      coef_text = "" if abs(mag - 1.0) < 1e-9 else _fmt_number(mag)
      terms.append(f"{sign}{coef_text}{var}")
    if not terms:
      left = "0"
    else:
      left = " ".join(terms).lstrip("+")
    parts.append(f"{left} {op} {_fmt_number(rhs)}")
  return " and ".join(parts)

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

def format_complex_solutions(solutions):
  if not solutions:
    return "no solution"
  return "{" + ", ".join(_fmt_complex(v) for v in solutions) + "}"

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

def _fmt_complex(value):
  real = value.real
  imag = value.imag
  if abs(imag) < 1e-9:
    return _fmt_number(real)
  has_real = abs(real) >= 1e-9
  real_text = _fmt_number(real) if has_real else ""
  imag_mag = abs(imag)
  imag_text = "" if abs(imag_mag - 1.0) < 1e-9 else _fmt_number(imag_mag)
  sign = "+" if imag >= 0 else "-"
  if not has_real:
    return f"{sign}{imag_text}i" if sign == "-" else f"{imag_text}i"
  return f"{real_text} {sign} {imag_text}i"

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

def solve_linear_system(equations):
  """
  Solves a linear system of equations with multiple variables.
  equations: list of strings like ["2x + y = 3", "x - y = 1"].
  Returns a dict mapping variable name to value.
  """
  if not equations:
    raise ValueError("No equations provided.")

  coeff_rows = []
  constants = []
  variables = set()

  for eq in equations:
    lhs, rhs = parsing.split_equation(eq)
    lhs_ast = parsing.parse_expr(lhs)
    rhs_ast = parsing.parse_expr(rhs)
    cL, kL = parsing.linearize_multi_ast(lhs_ast)
    cR, kR = parsing.linearize_multi_ast(rhs_ast)
    coeffs = {}
    for key, val in cL.items():
      coeffs[key] = coeffs.get(key, 0.0) + val
    for key, val in cR.items():
      coeffs[key] = coeffs.get(key, 0.0) - val
    const = kL - kR
    coeff_rows.append(coeffs)
    constants.append(-const)
    variables.update(coeffs.keys())

  vars_sorted = sorted(variables)
  n = len(vars_sorted)
  if len(coeff_rows) != n:
    raise ValueError("System must have the same number of equations as variables.")

  # Build augmented matrix.
  matrix = []
  for coeffs, const in zip(coeff_rows, constants):
    row = [coeffs.get(v, 0.0) for v in vars_sorted]
    row.append(const)
    matrix.append(row)

  # Gaussian elimination.
  for col in range(n):
    pivot = col
    for row in range(col, n):
      if abs(matrix[row][col]) > abs(matrix[pivot][col]):
        pivot = row
    if abs(matrix[pivot][col]) < 1e-12:
      raise ValueError("System is singular or underdetermined.")
    if pivot != col:
      matrix[col], matrix[pivot] = matrix[pivot], matrix[col]

    pivot_val = matrix[col][col]
    matrix[col] = [val / pivot_val for val in matrix[col]]

    for row in range(n):
      if row == col:
        continue
      factor = matrix[row][col]
      if abs(factor) < 1e-12:
        continue
      matrix[row] = [
        val - factor * matrix[col][i] for i, val in enumerate(matrix[row])
      ]

  solution = {var: matrix[i][-1] for i, var in enumerate(vars_sorted)}
  return solution
