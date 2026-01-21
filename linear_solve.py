from validation import normalize, split_terms

def reduce_linear(expr: str) -> tuple[float, float]:
    expr = normalize(expr)
    terms = split_terms(expr)

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
