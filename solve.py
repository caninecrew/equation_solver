import linear
import quadratic
import parsing
import polynomial
import transcendental
import nonlinear
from typing import cast

def solve(equation, xmin=None, xmax=None):
  """
  Solves equations, inequalities, or systems and returns formatted results.

  Args:
    equation (str | list[str] | tuple[str, ...]): Equation, inequality, or system.
    xmin (float, optional): Lower bound for numeric root/interval searches.
    xmax (float, optional): Upper bound for numeric root/interval searches.
  """
  if isinstance(equation, (list, tuple)):
          if not all(isinstance(e, str) for e in equation):
                  raise TypeError("All equations must be strings.")
          try:
                  solutions = linear.solve_linear_system(equation)
                  return "{" + ", ".join(f"{k}={v}" for k, v in solutions.items()) + "}"
          except ValueError:
                  solutions = nonlinear.solve_nonlinear_system_all(equation)
                  return "[" + ", ".join("{" + ", ".join(f"{k}={v}" for k, v in sol.items()) + "}" for sol in solutions) + "]"
  if isinstance(equation, str):
          # Expand chained equalities into a system.
          expanded = parsing.split_equalities(equation)
          if len(expanded) > 1:
                  return solve(expanded, xmin=xmin, xmax=xmax)
          if any(op in equation for op in ["<=", ">=", "<", ">"]):
                  try:
                          intervals = linear.solve_inequality(equation)
                          return linear.format_intervals(intervals)
                  except ValueError:
                          try:
                                  constraints = linear.solve_inequality_system(equation)
                                  return linear.format_halfspaces(constraints)
                          except ValueError:
                                  # Transcendental inequality fallback.
                                  try:
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
                                                  sign_intervals = transcendental.intervals_for_op(
                                                          expr_ast, op, xmin or -10.0, xmax or 10.0
                                                  )
                                                  intervals = linear._intersect_interval_lists(intervals, sign_intervals)
                                          return linear.format_intervals(intervals)
                                  except ValueError:
                                          pass

          # Try quadratic form first when possible.
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
          try:
                  coeffs = parsing.polynomialize_ast(expr_ast)
                  while len(coeffs) > 1 and abs(coeffs[-1]) < 1e-12:
                          coeffs.pop()
                  degree = len(coeffs) - 1
                  if degree >= 2:
                          if degree == 2:
                                  roots = quadratic.quadratic(coeffs[2], coeffs[1], coeffs[0])
                          else:
                                  roots = polynomial.roots(list(reversed(coeffs)))
                          if any(abs(r.imag) > 1e-7 for r in roots):
                                  return linear.format_complex_solutions(roots)
                          return linear.format_solutions([r.real for r in roots])
          except ValueError:
                  pass

          # Fallback to transcendental root finding.
          try:
                  roots = transcendental.find_real_roots(expr_ast, xmin or -10.0, xmax or 10.0)
                  return linear.format_solutions(roots)
          except ValueError:
                  pass

          # Final fallback: linear solver for single-variable equations.
          solutions = linear.solve_linear(equation)
          return linear.format_solutions(solutions)
  else:
      raise TypeError("Unsupported equation type")
