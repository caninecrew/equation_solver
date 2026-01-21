import math as m

from torch import fix_

def fix_zero(value, eps=1e-12):
    return 0.0 if abs(value) < eps else value

def quadratic(a, b, c):
  """
  Calculates the roots of a quadratic equation using the quadratic formula.

  Args:
    a (float): The coefficient of the x^2 term.
    b (float): The coefficient of the x term.
    c (float): The constant term.

  Returns:
    list[float]: A list containing the two roots of the quadratic equation.
  """
  # Calculate the discriminant (b^2 - 4ac)
  discriminant = (b**2) - (4*a*c)
  # Check for real roots
  if discriminant < 0: 
      return [] # No real roots
  sqr = m.sqrt(discriminant) # Calculate the square root of the discriminant
  bottom = 2*a # Calculate the denominator (2a)

  positive = (-b + sqr) / bottom # Calculate the first root using the '+' sign
  negative = (-b - sqr) / bottom # Calculate the second root using the '-' sign

  return [positive, negative] # Return both roots in a list

def equation_strip(lhs: str, rhs: str) -> tuple[str, str]:
    lhs = lhs.replace(" ", "")
    rhs = rhs.replace(" ", "")

    if lhs == "" or rhs == "":
        raise ValueError("Both sides of the equation must be non-empty.")

    combined = lhs + rhs

    if any(char not in "0123456789xX+-*/." for char in combined):
        raise ValueError("Equation contains invalid characters.")

    return lhs, rhs

def split_equation(equation: str) -> tuple[str, str]:
  """
  Splits an equation string into its left and right-hand sides.

  Args:
    equation (str): The full equation string (e.g., "2x + 3 = 7").

  Returns:
    tuple[str, str]: A tuple containing the left-hand side and right-hand side as strings.

  Raises:
    ValueError:
      - If the equation does not contain exactly one '=' sign.
      - If either side of the equation is empty after stripping whitespace.
  """
  eq = equation.strip() # Remove leading/trailing whitespace from the full equation

  if eq.count("=") != 1: # Check if there is exactly one '=' sign
    raise ValueError("Equation must contain exactly one '=' sign.") # Raise an error if not

  lhs, rhs = eq.split("=", 1) # Split the equation into left and right sides at the first '=' sign

  lhs = lhs.strip() # Remove leading/trailing whitespace from the left-hand side
  rhs = rhs.strip() # Remove leading/trailing whitespace from the right-hand side

  if lhs == "" or rhs == "": # Check if either side is empty after stripping
    raise ValueError("Both sides of the equation must be non-empty.") # Raise an error if empty

  return equation_strip(lhs, rhs) # Further strip and validate variables using equation_strip

def normalize(expr: str) -> str:
  """
  Normalizes an algebraic expression by removing spaces and adding a leading '+'
  if the first character is not a sign.

  Args:
    expr (str): The algebraic expression string.

  Returns:
    str: The normalized expression.
  """
  expr = expr.replace(" ", "") # Remove all spaces from the expression

  if expr[0] not in "+-": # If the first character is not a '+' or '-'
    expr = "+" + expr # Prepend a '+' to the expression

  return expr # Return the normalized expression

def split_terms(expr: str) -> list[str]:
  """
  Splits an algebraic expression into individual terms based on '+' or '-' signs.
  For example, "+2x-3" becomes ["+2x", "-3"].

  Args:
    expr (str): The algebraic expression string.

  Returns:
    list[str]: A list of terms, each including its leading sign.
  """
  terms = [] # Initialize an empty list to store terms
  start = 0 # Initialize the start index for slicing terms

  for i in range(1, len(expr)): # Iterate from the second character to the end
    if expr[i] in "+-": # If a '+' or '-' sign is found
      terms.append(expr[start:i]) # Append the term found so far
      start = i # Update the start index for the next term

  terms.append(expr[start:]) # Append the last term after the loop finishes
  return terms # Return the list of terms

def reduce_linear(expr):
  """
  Reduces a linear algebraic expression into the coefficients of 'x' and the constant term.

  Args:
    expr (str): The linear algebraic expression (e.g., "+2x+6").

  Returns:
    tuple[float, float]: A tuple containing the coefficient of 'x' (a) and the constant term (b).

  Raises:
    ValueError: If non-linear terms (e.g., 'x' appearing multiple times in a term) are found.
  """
  expr = normalize(expr) # Normalize the expression (e.g., add leading '+')
  terms = split_terms(expr) # Split the expression into individual terms

  a = 0.0 # Initialize coefficient of 'x'
  b = 0.0 # Initialize constant term

  for term in terms: # Iterate through each term
    term = term.replace("*", "") # Remove multiplication signs if present (e.g., 2*x becomes 2x)

    if "x" in term: # If 'x' is present in the term, it's an 'x' term
      if term.count("x") != 1: # Check if 'x' appears more than once in the term
        raise ValueError("Only linear terms are supported (like 2x, -x, +x).") # Raise error for non-linear 'x' terms

      coef_text = term.replace("x", "") # Remove 'x' to get the coefficient text

      if coef_text == "+" or coef_text == "": # Handle cases like '+x' or 'x'
        coef = 1.0
      elif coef_text == "-": # Handle case like '-x'
        coef = -1.0
      else:
        coef = float(coef_text) # Convert the coefficient text to a float

      a += coef # Add the coefficient to the total 'a' (coefficient of x)

    else: # If 'x' is not in the term, it's a constant term
      b += float(term) # Add the constant term to the total 'b'

  return a, b # Return the reduced coefficients (a for x, b for constant)

def solve_linear(equation):
  """
  Solves a linear equation of the form ax + b = 0.
  """

  lhs, rhs = split_equation(equation) # Split the equation into left and right sides
  aL, bL = reduce_linear(lhs) # Reduce the left-hand side to get 'a' and 'b'
  aR, bR = reduce_linear(rhs) # Reduce the right-hand side to get 'a' and 'b'

  a = aL - aR # Calculate the coefficient of 'x'
  b = bL - bR # Calculate the constant term

  if a != 0: # If the coefficient of 'x' is not zero
    return [fix_zero(-b / a)] # Return the single root

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

def solve(equation, xmin=None, xmax=None):
  """
  Placeholder function for solving an equation.
  This function is not yet implemented.

  Args:
    equation (str): The equation to solve.
    xmin (float, optional): Minimum value for the solution range. Defaults to None.
    xmax (float, optional): Maximum value for the solution range. Defaults to None.
  """
  if isinstance(equation, str): # Check if the equation is a string
          # Currently only linear equations are supported at this entry point.
          return solve_linear(equation) # If it's a string, attempt to solve it as a linear equation
  else:
      raise TypeError("Unsupported equation type") # Raise an error for unsupported equation types

print(solve("2x + 3 = 7")) # Example usage
print(solve("2x + 7 = 7"))