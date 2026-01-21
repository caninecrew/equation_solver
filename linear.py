import utils
import parsing

def reduce_linear(expr):
  """
  Reduces a linear algebraic expression into the coefficients of 'x' and the constant term.
  Supports terms like 4x, -x, +x, 3*x, and decimals.
  """
  expr = parsing.normalize(expr)
  terms = parsing.split_terms(expr)

  a = 0.0
  b = 0.0

  for term in terms:
    term = term.replace("*", "")

    if "x" in term:
      if term.count("x") != 1:
        raise ValueError("Only linear terms are supported (like 2x, -x, +x).")

      coef_text = term.replace("x", "")
      if coef_text == "+" or coef_text == "":
        coef = 1.0
      elif coef_text == "-":
        coef = -1.0
      else:
        coef = float(coef_text)

      a += coef
    else:
      b += float(term)

  return a, b

def solve_linear(equation):
  """
  Solves a linear equation of the form ax + b = 0.
  """

  lhs, rhs = parsing.split_equation(equation) # Split the equation into left and right sides
  aL, bL = reduce_linear(lhs) # Reduce the left-hand side to get 'a' and 'b'
  aR, bR = reduce_linear(rhs) # Reduce the right-hand side to get 'a' and 'b'

  a = aL - aR # Calculate the coefficient of 'x'
  b = bL - bR # Calculate the constant term

  if a != 0: # If the coefficient of 'x' is not zero
    return [utils.fix_zero(-b / a)] # Return the single root

  if b == 0: # If both 'a' and 'b' are zero (e.g., 0=0)
    return ["ALL_REAL_NUMBERS"] # The equation is an identity

  return [] # If 'a' is zero and 'b' is not zero (e.g., 0=5), there is no solution

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