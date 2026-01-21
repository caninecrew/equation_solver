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

def equation_strip(lhs: str, rhs: str) -> tuple[str, str]:
    lhs = lhs.replace(" ", "")
    rhs = rhs.replace(" ", "")

    if lhs == "" or rhs == "":
        raise ValueError("Both sides of the equation must be non-empty.")

    combined = lhs + rhs

    # Allow the function name "abs" by removing it before validation
    combined_without_abs = combined.replace("abs", "")

    # Now only 'x' is allowed as a remaining letter
    for ch in combined_without_abs:
        if ch.isalpha() and ch != "x":
            raise ValueError("Only the variable 'x' (and abs(...)) is supported in this solver.")

    return lhs, rhs

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
