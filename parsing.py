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

def split_inequality(equation: str) -> tuple[str, str, str]:
  """
  Splits an inequality string into its left-hand side, operator, and right-hand side.

  Args:
    equation (str): The full inequality string (e.g., "2x + 3 <= 7").

  Returns:
    tuple[str, str, str]: A tuple containing the left-hand side, operator, and right-hand side.

  Raises:
    ValueError:
      - If the inequality does not contain exactly one comparison operator.
      - If either side of the inequality is empty after stripping whitespace.
  """
  eq = equation.strip()
  match = re.search(r"(<=|>=|<|>)", eq)
  if not match:
    raise ValueError("Inequality must contain exactly one comparison operator.")
  op = match.group(1)
  if re.search(r"(<=|>=|<|>)", eq[match.end():]):
    raise ValueError("Inequality must contain exactly one comparison operator.")

  lhs, rhs = eq.split(op, 1)
  lhs = lhs.strip()
  rhs = rhs.strip()

  if lhs == "" or rhs == "":
    raise ValueError("Both sides of the inequality must be non-empty.")

  lhs, rhs = equation_strip(lhs, rhs)
  return lhs, op, rhs

def equation_strip(lhs: str, rhs: str) -> tuple[str, str]:

    lhs = lhs.replace(" ", "")
    rhs = rhs.replace(" ", "")

    if lhs == "" or rhs == "":
        raise ValueError("Both sides of the equation must be non-empty.")

    combined = lhs + rhs

    if any(not (char.isalnum() or char in "+-*/.()^") for char in combined):
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

def parse_expr(expr: str) -> ast.expr:
    """
    Parses an algebraic expression string into an AST node.

    Args:
        expr (str): The algebraic expression string.

    Returns:
        ast.AST: The root AST node representing the expression.
    """
    expr = insert_implicit_mul(expr)  # optional but recommended
    expr = expr.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return tree.body

def insert_implicit_mul(expr: str) -> str:
    """
    Inserts explicit multiplication operators in an algebraic expression where multiplication is implied (e.g., between a number and a variable).

    Args:
        expr (str): The algebraic expression string.

    Returns:
        str: The expression with explicit multiplication operators.
    """

    expr = expr.replace(" ", "")
    expr = re.sub(r'(\d)([xX(])', r'\1*\2', expr)
    expr = re.sub(r'([xX)])(\d|\()', r'\1*\2', expr)
    expr = re.sub(r'(\))([xX(])', r'\1*\2', expr)
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
    
    # Handle variable 'x'
    if isinstance(node, ast.Name):
        if node.id != "x":
            raise ValueError("Only 'x' is supported.")
        return 1.0, 0.0

    # Handle unary operations
    if isinstance(node, ast.UnaryOp):
        a, b = linearize_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -a, -b
        if isinstance(node.op, ast.UAdd):
            return a, b
        raise ValueError("Unsupported unary operator.")
    
    # Handle function calls (only abs supported)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "abs":
            raise ValueError("Only abs() is supported.")
        if len(node.args) != 1:
            raise ValueError("abs() takes exactly one argument.")
        a, b = linearize_ast(node.args[0])
        if a != 0.0:
            raise ValueError("abs() with x is not supported for linear solver.")
        return 0.0, abs(b)

    # Handle binary operations
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
        if isinstance(node.op, ast.Pow):
            if a1 != 0.0 or a2 != 0.0:
                raise ValueError("Exponentiation with x is not supported.")
            return 0.0, b1 ** b2

        raise ValueError("Unsupported binary operator.")
    raise ValueError("Unsupported expression node.")

def quadraticize_ast(node) -> tuple[float, float, float]:
    """
    Converts an AST node into quadratic coefficients (a, b, c) for a*x^2 + b*x + c.
    Supports +, -, *, / with constants and x, and x^2.
    """
    def add_poly(p, q):
        return (p[0] + q[0], p[1] + q[1], p[2] + q[2])

    def sub_poly(p, q):
        return (p[0] - q[0], p[1] - q[1], p[2] - q[2])

    def mul_poly(p, q):
        # (c0 + c1*x + c2*x^2) * (d0 + d1*x + d2*x^2)
        c0, c1, c2 = p
        d0, d1, d2 = q
        # Compute up to degree 2, reject higher-degree terms.
        deg3 = c1 * d2 + c2 * d1
        deg4 = c2 * d2
        if deg3 != 0.0 or deg4 != 0.0:
            raise ValueError("Polynomial degree greater than 2 is not supported.")
        return (
            c0 * d0,
            c0 * d1 + c1 * d0,
            c0 * d2 + c1 * d1 + c2 * d0,
        )

    def div_poly(p, q):
        # Only allow division by constant.
        if q[1] != 0.0 or q[2] != 0.0:
            raise ValueError("Division by non-constant is not supported.")
        if q[0] == 0.0:
            raise ZeroDivisionError("Division by zero.")
        return (p[0] / q[0], p[1] / q[0], p[2] / q[0])

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants supported.")
        return (float(node.value), 0.0, 0.0)

    if isinstance(node, ast.Name):
        if node.id != "x":
            raise ValueError("Only 'x' is supported.")
        return (0.0, 1.0, 0.0)

    if isinstance(node, ast.UnaryOp):
        a, b, c = quadraticize_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return (-a, -b, -c)
        if isinstance(node.op, ast.UAdd):
            return (a, b, c)
        raise ValueError("Unsupported unary operator.")

    if isinstance(node, ast.BinOp):
        left = quadraticize_ast(node.left)
        right = quadraticize_ast(node.right)

        if isinstance(node.op, ast.Add):
            return add_poly(left, right)
        if isinstance(node.op, ast.Sub):
            return sub_poly(left, right)
        if isinstance(node.op, ast.Mult):
            return mul_poly(left, right)
        if isinstance(node.op, ast.Div):
            return div_poly(left, right)
        if isinstance(node.op, ast.Pow):
            # Only allow x^2 or constant^2.
            if right[1] != 0.0 or right[2] != 0.0:
                raise ValueError("Exponent must be constant.")
            exp = right[0]
            if exp == 2:
                return mul_poly(left, left)
            if exp == 1:
                return left
            if exp == 0:
                return (1.0, 0.0, 0.0)
            raise ValueError("Only exponent 2 is supported for x.")

        raise ValueError("Unsupported binary operator.")

    raise ValueError("Unsupported expression node.")

def reduce_linear(expr: str) -> tuple[float, float]:
    '''
    Reduces a linear algebraic expression into the coefficients of 'x' and the constant term.
    Supports terms like 4x, -x, +x, 3*x, and decimals.

    Args:
        expr (str): The algebraic expression string.
    
    Returns:
        tuple[float, float]: 
            A tuple (a, b) where 'a' is the coefficient of 'x' and 'b' is the constant term. 
    '''
    node = parse_expr(expr)
    return linearize_ast(node)

def find_abs_calls(node):
    """
    Finds all abs() function calls in an AST node. This is useful for identifying absolute value expressions in an equation and includes their arguments for further processing.

    Args:
        node: An AST node representing the expression.
    Returns:
        list: A list of AST nodes that are arguments to abs() function calls.
    """
    found = []

    def visit(n):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id == "abs" and len(n.args) == 1:
                found.append(n.args[0])
            # keep walking call args too
            for arg in n.args:
                visit(arg)
            return

        # Walk binary/unary/other expression nodes
        if isinstance(n, ast.BinOp):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, ast.UnaryOp):
            visit(n.operand)
        elif isinstance(n, ast.BoolOp):
            for v in n.values:
                visit(v)
        elif isinstance(n, ast.Compare):
            visit(n.left)
            for c in n.comparators:
                visit(c)
        elif isinstance(n, ast.IfExp):
            visit(n.test)
            visit(n.body)
            visit(n.orelse)
        elif isinstance(n, ast.Call):
            # handled above
            pass
        # constants/names: nothing to do

    visit(node)
    return found

def build_abs_cases(expr_ast):
    """
    Expands abs() into piecewise cases. Returns a list of (expr_ast, constraints).
    Each constraints entry is a list of Compare nodes to evaluate at a candidate x.
    """
    def expand(node):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "abs":
                raise ValueError("Only abs() calls are supported.")
            if len(node.args) != 1:
                raise ValueError("abs() takes one argument.")
            inner_cases = expand(node.args[0])
            cases = []
            for inner_expr, inner_constraints in inner_cases:
                pos_constraint = ast.Compare(
                    left=inner_expr, ops=[ast.GtE()], comparators=[ast.Constant(0)]
                )
                neg_constraint = ast.Compare(
                    left=inner_expr, ops=[ast.Lt()], comparators=[ast.Constant(0)]
                )
                cases.append((inner_expr, inner_constraints + [pos_constraint]))
                cases.append(
                    (ast.UnaryOp(op=ast.USub(), operand=inner_expr), inner_constraints + [neg_constraint])
                )
            return cases

        if isinstance(node, ast.BinOp):
            left_cases = expand(node.left)
            right_cases = expand(node.right)
            cases = []
            for left_expr, left_constraints in left_cases:
                for right_expr, right_constraints in right_cases:
                    cases.append(
                        (
                            ast.BinOp(left=left_expr, op=node.op, right=right_expr),
                            left_constraints + right_constraints,
                        )
                    )
            return cases

        if isinstance(node, ast.UnaryOp):
            operand_cases = expand(node.operand)
            return [
                (ast.UnaryOp(op=node.op, operand=expr), constraints)
                for expr, constraints in operand_cases
            ]

        if isinstance(node, ast.Constant) or isinstance(node, ast.Name):
            return [(node, [])]

        raise ValueError("Unsupported expression node for abs expansion.")

    return expand(expr_ast)

def eval_linear_ast(node, x_value):
    # returns numeric value of the expression at x
    if isinstance(node, ast.Constant):
        value = node.value
        if not isinstance(value, (int, float)):
            raise ValueError("Only numeric constants supported.")
        return float(value)
    if isinstance(node, ast.Name):
        if node.id != "x":
            raise ValueError("Only 'x' is supported.")
        return float(x_value)
    if isinstance(node, ast.UnaryOp):
        val = eval_linear_ast(node.operand, x_value)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("Unsupported unary op.")
    if isinstance(node, ast.BinOp):
        left = eval_linear_ast(node.left, x_value)
        right = eval_linear_ast(node.right, x_value)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow): return left ** right
        raise ValueError("Unsupported binary op.")
    raise ValueError("Unsupported node.")

def eval_constraint(node, x_value):
    if not isinstance(node, ast.Compare):
        raise ValueError("Expected Compare node.")
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise ValueError("Only single comparisons supported.")
    left = eval_linear_ast(node.left, x_value)
    right = eval_linear_ast(node.comparators[0], x_value)
    op = node.ops[0]
    if isinstance(op, ast.GtE): return left >= right
    if isinstance(op, ast.Lt): return left < right
    if isinstance(op, ast.Gt): return left > right
    if isinstance(op, ast.LtE): return left <= right
    if isinstance(op, ast.Eq): return left == right
    raise ValueError("Unsupported comparison.")
