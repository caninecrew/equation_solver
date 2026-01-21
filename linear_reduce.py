"""
Linear reducer module.

Turns a linear expression string into (a, b) coefficients for a*x + b.
"""

from validation import normalize, split_terms

def reduce_linear(expr: str) -> tuple[float, float]:
    """
    Reduce a linear expression into coefficients.

    Args:
        expr: Expression like "2x-3" or "-x+5".

    Returns:
        (a, b) where the expression is equivalent to a*x + b.
    """
    # Normalize signs and spacing for consistent parsing.
    expr = normalize(expr)
    terms = split_terms(expr)

    a = 0.0
    b = 0.0

    for term in terms:
        # Remove explicit multiplication for easier parsing (e.g., "2*x").
        term = term.replace("*", "")
        if "x" in term:
            # Ensure the term is linear (only one x).
            if term.count("x") != 1:
                raise ValueError("Only linear terms are supported (like 2x, -x, +x).")
            # Extract the coefficient text by removing the variable.
            coef_text = term.replace("x", "")
            if coef_text == "+" or coef_text == "":
                coef = 1.0
            elif coef_text == "-":
                coef = -1.0
            else:
                coef = float(coef_text)
            a += coef
        else:
            # Constant term.
            b += float(term)

    return a, b
