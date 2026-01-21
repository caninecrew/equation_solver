import math
import parsing

def _sign(value):
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0

def _bisect(f, a, b, fa, fb, tol=1e-8, max_iter=100):
  if fa == 0:
    return a
  if fb == 0:
    return b
  for _ in range(max_iter):
    mid = (a + b) / 2.0
    try:
      fm = f(mid)
    except ValueError:
      # Domain error: shrink interval.
      mid = (a + mid) / 2.0
      try:
        fm = f(mid)
      except ValueError:
        mid = (mid + b) / 2.0
        fm = f(mid)
    if abs(fm) <= tol:
      return mid
    if _sign(fa) * _sign(fm) < 0:
      b = mid
      fb = fm
    else:
      a = mid
      fa = fm
  return (a + b) / 2.0

def find_real_roots(expr_ast, xmin=-10.0, xmax=10.0, steps=400):
  def f(x):
    return parsing.eval_expr_ast(expr_ast, x)

  roots = []
  step = (xmax - xmin) / steps
  x0 = xmin
  try:
    f0 = f(x0)
  except ValueError:
    f0 = float("nan")
  if f0 == 0:
    roots.append(x0)
  for i in range(1, steps + 1):
    x1 = xmin + i * step
    try:
      f1 = f(x1)
    except ValueError:
      f1 = float("nan")
    if f1 == 0:
      roots.append(x1)
    if math.isfinite(f0) and math.isfinite(f1) and _sign(f0) * _sign(f1) < 0:
      root = _bisect(f, x0, x1, f0, f1)
      roots.append(root)
    x0, f0 = x1, f1
  return roots
