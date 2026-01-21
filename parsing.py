import re # Import the regular expressions module
import ast # Import the abstract syntax tree module

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

    if any(char not in "0123456789xX+-*/." for char in combined):
        raise ValueError("Equation contains invalid characters.")

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

def insert_implicit_mul(expr: str) -> str:
    """
    Inserts explicit multiplication operators '*' where implicit multiplication is detected.
    For example, "2x" becomes "2*x" and "3(x + 1)" becomes "3*(x + 1)".

    Args:
        expr (str): The algebraic expression string.

    Returns:
        str: The expression with explicit multiplication operators.
    """

    expr = expr.replace(" ", "")
    # number followed by symbol or '('
    expr = re.sub(r'(\d)([A-Za-z(])', r'\1*\2', expr)
    # symbol or ')' followed by number or '('
    expr = re.sub(r'([A-Za-z)])(\d|\()', r'\1*\2', expr)
    # ')' followed by symbol or '('
    expr = re.sub(r'(\))([A-Za-z(])', r'\1*\2', expr)
    return expr

def linearize_ast(node) -> tuple[float, float]:
    """
    Converts an AST node representing a linear expression into its coefficients (a, b) for the form a*x + b.

    Args:
        node: An AST node representing the expression.
    Returns:
        tuple[float, float]:
            A tuple (a, b) representing the coefficients of the linear expression a*x + b.
    """
    # returns (a, b) for a*x + b
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants supported.")
        return 0.0, float(node.value)

    if isinstance(node, ast.Name):
        if node.id != "x":
            raise ValueError("Only 'x' is supported.")
        return 1.0, 0.0

    if isinstance(node, ast.UnaryOp):
        a, b = linearize_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -a, -b
        if isinstance(node.op, ast.UAdd):
            return a, b
        raise ValueError("Unsupported unary operator.")

    if isinstance(node, ast.BinOp):
        a1, b1 = linearize_ast(node.left)
        a2, b2 = linearize_ast(node.right)

        if isinstance(node.op, ast.Add):
            return a1 + a2, b1 + b2
        if isinstance(node.op, ast.Sub):
            return a1 - a2, b1 - b2
        if isinstance(node.op, ast.Mult):
            # nonlinear if both sides have x
            if a1 != 0.0 and a2 != 0.0:
                raise ValueError("Nonlinear term in multiplication.")
            if a2 == 0.0:
                return a1 * b2, b1 * b2
            return a2 * b1, b2 * b1
        if isinstance(node.op, ast.Div):
            if a2 != 0.0:
                raise ValueError("Nonlinear term in division.")
            if b2 == 0.0:
                raise ZeroDivisionError("Division by zero.")
            return a1 / b2, b1 / b2

        raise ValueError("Unsupported binary operator.")
    raise ValueError("Unsupported expression node.")