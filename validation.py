def equation_strip(lhs: str, rhs: str) -> tuple[str, str]:
    lhs = lhs.replace(" ", "")
    rhs = rhs.replace(" ", "")
    if lhs == "" or rhs == "":
        raise ValueError("Both sides of the equation must be non-empty.")

    combined = lhs + rhs
    combined_without_abs = combined.replace("abs", "")
    for ch in combined_without_abs:
        if ch.isalpha() and ch != "x":
            raise ValueError("Only the variable 'x' (and abs(...)) is supported.")
    return lhs, rhs

def split_equation(equation: str) -> tuple[str, str]:
    eq = equation.strip()
    if eq.count("=") != 1:
        raise ValueError("Equation must contain exactly one '=' sign.")
    lhs, rhs = eq.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if lhs == "" or rhs == "":
        raise ValueError("Both sides of the equation must be non-empty.")
    return equation_strip(lhs, rhs)

def normalize(expr: str) -> str:
    expr = expr.replace(" ", "")
    if expr[0] not in "+-":
        expr = "+" + expr
    return expr

def split_terms(expr: str) -> list[str]:
    terms = []
    start = 0
    for i in range(1, len(expr)):
        if expr[i] in "+-":
            terms.append(expr[start:i])
            start = i
    terms.append(expr[start:])
    return terms
