import utils
import parsing

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