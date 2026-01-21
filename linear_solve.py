"""
Linear solving module.

Note: This file currently contains reduce_linear; if you split modules,
move reduce_linear into linear_reduce.py and keep only solve functions here.
"""

from validation import normalize, split_terms

def reduce_linear(expr: str) -> tuple[float, float]:
    """
    Reduce a linear expression into (a, b) coefficients.

    Args:
        expr: Expression like "2x-3".

    Returns:
        (a, b) for a*x + b.
    """
    # Normalize signs and remove spaces for predictable parsing.
    expr = normalize(expr)
    terms = split_terms(expr)

    a = 0.0
    b = 0.0

    for term in terms:
        # Remove explicit multiplication (e.g., "2*x" -> "2x").
        term = term.replace("*", "")
        if "x" in term:
            # Ensure only one x appears in a linear term.
            if term.count("x") != 1:
                raise ValueError("Only linear terms are supported (like 2x, -x, +x).")
            # The remaining text is the coefficient.
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
