Add a parser helper to find abs() calls in the AST and return the inner node.
Create a function to build two case expressions: abs(f) → +f with constraint f>=0, and -f with constraint f<0.
Update linearize_ast to allow abs() nodes to pass through (or handle them in a separate preprocessing step).
In solve_linear, for each case: solve the linear equation, then evaluate the constraint at the solution and filter invalid ones.
Add a small helper to evaluate a linear AST at a specific x to check constraints.
Add tests for abs(x+1)=3, abs(2x-4)=0, and a case with no valid solutions.