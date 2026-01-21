Equation Solver
==============

A general-purpose equation and inequality solver with support for linear,
polynomial, transcendental, and multi-variable systems. The solver prioritizes
clarity and practical numeric results over symbolic algebra.

Capabilities
------------

Single-variable equations
~~~~~~~~~~~~~~~~~~~~~~~~~
- Linear equations (exact).
- Polynomial equations of any degree (numeric roots).
- Quadratic equations (numeric roots).
- Transcendental equations using `sin`, `cos`, `tan`, `log`, `exp`, `sqrt`,
  and `abs` (numeric roots).

Inequalities
~~~~~~~~~~~~
- Linear inequalities with interval output.
- Chained inequalities (e.g., `1 < x < 3`).
- Multi-variable linear inequalities return half-space constraints.

Multi-variable systems
~~~~~~~~~~~~~~~~~~~~~~
- Linear systems (Gaussian elimination).
- Nonlinear systems (numeric Newton method with multi-start).

Piecewise expressions
~~~~~~~~~~~~~~~~~~~~~
- `piecewise(cond, expr, ..., default)`
- Python-style ternary: `expr if cond else expr`

Complex roots
~~~~~~~~~~~~~
- Polynomial roots may include complex solutions (formatted using `i`).

Usage
-----

```python
from solve import solve

# Linear equation
print(solve("2x + 3 = 7"))

# Linear inequalities
print(solve("1 < x < 3"))
print(solve("0 <= x <= 2 - x"))

# Polynomials (any degree)
print(solve("x^3 - 6x^2 + 11x - 6 = 0"))

# Transcendental equation (numeric range)
print(solve("sin(x) - 0.5 = 0", xmin=0, xmax=10))

# Complex roots
print(solve("x^2 + 1 = 0"))

# Linear system (multi-variable)
print(solve(["2x + y = 3", "x - y = 1"]))

# Nonlinear system (multi-variable)
print(solve(["x^2 + y^2 = 1", "x - y = 0"]))

# Piecewise
print(solve("piecewise(x<0, -x, x>=0, x, 0) = 1"))
```

Output Formats
--------------

Equations
~~~~~~~~~
- Single-variable solutions: `{value}` or `{value1, value2}`.
- All real numbers: `all real numbers`.
- No solution: `no solution`.
- Complex roots: `{a + bi, a - bi}`.
- Linear systems: `{x=2, y=1}`.
- Nonlinear systems: `[{x=0.7, y=0.7}, {x=-0.7, y=-0.7}]`.

Inequalities
~~~~~~~~~~~~
Intervals are formatted as:
- `(a, b)` for open intervals
- `[a, b]` for closed intervals
- Mixed endpoints for half-open intervals
- Unbounded intervals use `-∞` and `∞`
- Full-space and empty results use `all real numbers` or `no solution`

Multi-variable inequalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Return a conjunction of half-space constraints, e.g.:

```
x + y > 1 and x + y < 4
```

Notes and Limitations
---------------------
- Transcendental solving is numeric and may miss roots with even multiplicity
  or flat zero crossings.
- Nonlinear systems use Newton's method and may fail to converge depending on
  the initial guesses; a failure raises `ValueError`.
- Inequalities with nonlinear multi-variable expressions are not fully solved
  as regions; only linear half-spaces are supported.
- Domain restrictions for `log` and `sqrt` are enforced during evaluation.

Additional Examples
-------------------

```python
# Linear equation
solve("3x - 9 = 0")          # {3}

# Linear inequality
solve("2x + 1 >= 5")         # [2, ∞)

# Chained inequality with expression
solve("1 < x+2 < 4")         # (-1, 2)

# Polynomial (degree 4)
solve("x^4 - 5x^2 + 4 = 0")  # {-2, -1, 1, 2} (numeric approximations)

# Complex roots
solve("x^2 + 4 = 0")         # {2i, -2i}

# Transcendental equation
solve("cos(x) = 0", xmin=0, xmax=10)

# Rational equation
solve("1/(x+1) = 2", xmin=-10, xmax=10)  # {-0.5}

# Piecewise equation
solve("piecewise(x<0, -x, x>=0, x, 0) = 2")  # {-2, 2}

# Linear system
solve(["x + y = 4", "x - y = 2"])        # {x=3, y=1}

# Nonlinear system
solve(["x^2 + y^2 = 1", "x - y = 0"])    # {x=0.7071, y=0.7071} or {x=-0.7071, y=-0.7071}

# Multi-variable inequality (half-spaces)
solve("x + y <= 3")                      # x + y <= 3
solve("1 < x + y < 4")                   # x + y > 1 and x + y < 4
```

Install
-------
This is a local project. Use the solver by importing `solve` or running the
module directly.

Command-line (module entry point):

```bash
python __main__.py
```

The current `__main__.py` runs a few example solves from the repo root.

Supported Input
---------------
- Equations: `lhs = rhs`
- Chained equalities: `x = y = z` (expanded into a system)
- Inequalities: `<`, `<=`, `>`, `>=` with optional chaining
- Variables: any identifier; single-variable problems are normalized to `x`
- Functions: `abs(x)`, `sin(x)`, `cos(x)`, `tan(x)`, `log(x)`, `exp(x)`,
  `sqrt(x)`, `piecewise(cond, expr, ..., default)`
- Implicit multiplication: `2x`, `3(x+1)`, `(x+1)(x-1)`

Web (Pyodide)
-------------
For GitHub Pages or other static hosting, use `web_bundle.py` to run the solver
in the browser with Pyodide.

Basic usage (Pyodide):

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
<script>
async function loadSolver() {
  const pyodide = await loadPyodide();
  const response = await fetch("web_bundle.py");
  const source = await response.text();
  pyodide.runPython(source);
  const solve = pyodide.globals.get("solve");
  console.log(solve("2x + 3 = 7"));
}
loadSolver();
</script>
```

Project Layout
--------------
- `solve.py`: main entry point and dispatcher.
- `parsing.py`: expression parsing and AST utilities.
- `linear.py`: linear solvers and inequality handling.
- `quadratic.py`, `polynomial.py`: polynomial solvers.
- `transcendental.py`: numeric root finding for non-polynomial expressions.
- `nonlinear.py`: nonlinear system solver.
- `web_bundle.py`: browser-ready bundle for Pyodide.
