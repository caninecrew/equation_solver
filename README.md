# Equation Solver

An extensible Python-based equation solver focused on algebra. Current support includes single-variable linear equations and absolute value equations, with an architecture intended to grow toward expression-based solving.

## Features
- Parses single-variable equations involving `x`
- Validates and normalizes equation strings
- Solves linear equations of the form `ax + b = 0`
  - One solution
  - No solution
  - Infinitely many solutions
- Solves absolute value equations:
  - `abs(A) = B` where `B` is a constant
  - `abs(A) = abs(C)` using algebraic identities
- Reuses coefficient-based algebra and deduplicates solution sets

## Project Goals
- Separate parsing/validation, expression reduction, and solving
- Avoid brittle string-based algebra where possible
- Keep solvers composable (absolute value reduces to linear cases)
- Enable future expansion without rewrites

## Planned Extensions
- Parentheses and distribution (e.g., `2(x+3)=18`)
- Quadratic equations (`ax^2 + bx + c`)
- Inequalities
- Numeric solvers as a fallback
- Migration toward an internal expression/AST representation

## Usage
Run the module and call the solving functions directly.

```python
from math import solve_linear, solve_absolute

print(solve_linear("2x+3=7"))           # [2.0]
print(solve_absolute("abs(x-3)=5"))     # [-2.0, 8.0]
```

## Notes
- The primary entry point will evolve toward a central dispatcher as the
  module structure expands.
