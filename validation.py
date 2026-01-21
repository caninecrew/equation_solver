"""
Validation and normalization utilities.

Provides string checks and helpers for equation parsing. We avoid strict
left/right semantics by treating equations as two unordered sides
("side_a", "side_b") that are later normalized into a canonical form.
"""

def equation_strip(lhs: str, rhs: str) -> tuple[str, str]:
    """
    Strip spaces and validate allowed symbols/variables on both sides.
    """
    # Remove spaces to simplify later parsing.
    lhs = lhs.replace(" ", "")
    rhs = rhs.replace(" ", "")
    if lhs == "" or rhs == "":
        raise ValueError("Both sides of the equation must be non-empty.")

    # Validate that only 'x' is used as a variable name.
    combined = lhs + rhs
    combined_without_abs = combined.replace("abs", "")
    for ch in combined_without_abs:
        if ch.isalpha() and ch != "x":
            raise ValueError("Only the variable 'x' (and abs(...)) is supported.")
    return lhs, rhs

def get_sides(equation: str) -> tuple[str, str]:
    """
    Split an equation into two sides without implying "left" or "right".
    """
    # Remove surrounding whitespace first.
    eq = equation.strip()
    if eq.count("=") != 1:
        raise ValueError("Equation must contain exactly one '=' sign.")
    side_a, side_b = eq.split("=", 1)
    side_a = side_a.strip()
    side_b = side_b.strip()
    if side_a == "" or side_b == "":
        raise ValueError("Both sides of the equation must be non-empty.")
    return equation_strip(side_a, side_b)

def split_equation(equation: str) -> tuple[str, str]:
    """
    Legacy wrapper for equation splitting.

    Prefer get_sides() to avoid left/right assumptions in future solvers.
    """
    return get_sides(equation)

def normalize(expr: str) -> str:
    """
    Normalize an expression to start with an explicit sign.
    """
    # Remove spaces for consistent term splitting.
    expr = expr.replace(" ", "")
    if expr[0] not in "+-":
        expr = "+" + expr
    return expr

def split_terms(expr: str) -> list[str]:
    """
    Split a signed expression into signed terms.
    """
    terms = []
    start = 0
    # Scan for sign changes to split terms.
    for i in range(1, len(expr)):
        if expr[i] in "+-":
            terms.append(expr[start:i])
            start = i
    terms.append(expr[start:])
    return terms
