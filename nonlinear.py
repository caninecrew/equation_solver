import parsing

def _solve_linear_system(matrix, tol=1e-12):
  n = len(matrix)
  for col in range(n):
    pivot = col
    for row in range(col, n):
      if abs(matrix[row][col]) > abs(matrix[pivot][col]):
        pivot = row
    if abs(matrix[pivot][col]) < tol:
      raise ValueError("Jacobian is singular.")
    if pivot != col:
      matrix[col], matrix[pivot] = matrix[pivot], matrix[col]

    pivot_val = matrix[col][col]
    matrix[col] = [val / pivot_val for val in matrix[col]]

    for row in range(n):
      if row == col:
        continue
      factor = matrix[row][col]
      if abs(factor) < tol:
        continue
      matrix[row] = [
        val - factor * matrix[col][i] for i, val in enumerate(matrix[row])
      ]
  return [row[-1] for row in matrix]

def solve_nonlinear_system(equations, max_iter=50, tol=1e-8, guess=None, scan=(-2.0, 2.0), scan_steps=3):
  if not equations:
    raise ValueError("No equations provided.")

  expr_asts = []
  variables = set()
  for eq in equations:
    lhs, rhs = parsing.split_equation(eq)
    lhs_ast = parsing.parse_expr(lhs)
    rhs_ast = parsing.parse_expr(rhs)
    expr_ast = parsing.ast.BinOp(
      left=lhs_ast,
      op=parsing.ast.Sub(),
      right=rhs_ast,
    )
    expr_asts.append(expr_ast)
    variables.update(parsing.get_variable_names(expr_ast))

  vars_sorted = sorted(variables)
  n = len(vars_sorted)
  if len(expr_asts) != n:
    raise ValueError("System must have the same number of equations as variables.")

  guesses = []
  if guess is not None:
    guesses.append([float(guess.get(v, 0.0)) for v in vars_sorted])
  else:
    lo, hi = scan
    if scan_steps < 1:
      scan_steps = 1
    if n == 1:
      for i in range(scan_steps + 1):
        t = lo + (hi - lo) * (i / scan_steps)
        guesses.append([t])
    else:
      grid = [lo + (hi - lo) * (i / scan_steps) for i in range(scan_steps + 1)]
      for i in range(len(grid)):
        for j in range(len(grid)):
          guesses.append([grid[i], grid[j]])

  def fvec(xvec):
    env = {v: xvec[i] for i, v in enumerate(vars_sorted)}
    return [parsing.eval_expr_ast_with_env(expr, env) for expr in expr_asts]

  for start in guesses:
    x = start[:]
    for _ in range(max_iter):
      try:
        fx = fvec(x)
      except ValueError:
        break
      norm = sum(abs(v) for v in fx)
      if norm < tol:
        return {v: x[i] for i, v in enumerate(vars_sorted)}

      # Jacobian via finite differences.
      jacobian = []
      for i in range(n):
        row = []
        for j in range(n):
          h = 1e-6 * (1.0 + abs(x[j]))
          xph = x[:]
          xph[j] += h
          try:
            fph = fvec(xph)[i]
          except ValueError:
            fph = fx[i]
          row.append((fph - fx[i]) / h)
        row.append(-fx[i])
        jacobian.append(row)

      try:
        delta = _solve_linear_system(jacobian)
      except ValueError:
        break

      # Damped update.
      step = 1.0
      improved = False
      for _ in range(10):
        trial = [x[i] + step * delta[i] for i in range(n)]
        try:
          trial_norm = sum(abs(v) for v in fvec(trial))
        except ValueError:
          trial_norm = float("inf")
        if trial_norm < norm:
          x = trial
          improved = True
          break
        step *= 0.5
      if not improved:
        break

  raise ValueError("Nonlinear solver did not converge.")

def solve_nonlinear_system_all(equations, max_iter=50, tol=1e-8, scan=(-2.0, 2.0), scan_steps=3, eps=1e-6):
  solutions = []
  expr_asts = []
  variables = set()
  for eq in equations:
    lhs, rhs = parsing.split_equation(eq)
    lhs_ast = parsing.parse_expr(lhs)
    rhs_ast = parsing.parse_expr(rhs)
    expr_ast = parsing.ast.BinOp(
      left=lhs_ast,
      op=parsing.ast.Sub(),
      right=rhs_ast,
    )
    expr_asts.append(expr_ast)
    variables.update(parsing.get_variable_names(expr_ast))

  vars_sorted = sorted(variables)
  n = len(vars_sorted)
  if len(expr_asts) != n:
    raise ValueError("System must have the same number of equations as variables.")

  lo, hi = scan
  grid = [lo + (hi - lo) * (i / scan_steps) for i in range(scan_steps + 1)]
  guesses = []
  if n == 1:
    for t in grid:
      guesses.append([t])
  else:
    for i in range(len(grid)):
      for j in range(len(grid)):
        guesses.append([grid[i], grid[j]])

  def fvec(xvec):
    env = {v: xvec[i] for i, v in enumerate(vars_sorted)}
    return [parsing.eval_expr_ast_with_env(expr, env) for expr in expr_asts]

  for start in guesses:
    x = start[:]
    for _ in range(max_iter):
      try:
        fx = fvec(x)
      except ValueError:
        break
      norm = sum(abs(v) for v in fx)
      if norm < tol:
        sol = {v: x[i] for i, v in enumerate(vars_sorted)}
        if not any(all(abs(sol[k] - s[k]) < eps for k in sol) for s in solutions):
          solutions.append(sol)
        break

      jacobian = []
      for i in range(n):
        row = []
        for j in range(n):
          h = 1e-6 * (1.0 + abs(x[j]))
          xph = x[:]
          xph[j] += h
          try:
            fph = fvec(xph)[i]
          except ValueError:
            fph = fx[i]
          row.append((fph - fx[i]) / h)
        row.append(-fx[i])
        jacobian.append(row)

      try:
        delta = _solve_linear_system(jacobian)
      except ValueError:
        break

      step = 1.0
      improved = False
      for _ in range(10):
        trial = [x[i] + step * delta[i] for i in range(n)]
        try:
          trial_norm = sum(abs(v) for v in fvec(trial))
        except ValueError:
          trial_norm = float("inf")
        if trial_norm < norm:
          x = trial
          improved = True
          break
        step *= 0.5
      if not improved:
        break

  if not solutions:
    raise ValueError("Nonlinear solver did not converge.")
  return solutions
