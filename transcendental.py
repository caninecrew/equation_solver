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
    except (ValueError, ZeroDivisionError):
      # Domain error: shrink interval.
      mid = (a + mid) / 2.0
      try:
        fm = f(mid)
      except (ValueError, ZeroDivisionError):
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

def _derivative(f, x, h=1e-5):
  return (f(x + h) - f(x - h)) / (2 * h)

def _newton(f, x0, tol=1e-10, max_iter=50):
  x = x0
  for _ in range(max_iter):
    fx = f(x)
    if abs(fx) <= tol:
      return x
    dfx = _derivative(f, x)
    if dfx == 0:
      return x
    x -= fx / dfx
  return x

def find_real_roots(expr_ast, xmin=-10.0, xmax=10.0, steps=400, zero_eps=1e-6):
  def f(x):
    return parsing.eval_expr_ast(expr_ast, x)

  roots = []
  step = (xmax - xmin) / steps
  x0 = xmin
  try:
    f0 = f(x0)
  except (ValueError, ZeroDivisionError):
    f0 = float("nan")
  if f0 == 0:
    roots.append(x0)
  for i in range(1, steps + 1):
    x1 = xmin + i * step
    try:
      f1 = f(x1)
    except (ValueError, ZeroDivisionError):
      f1 = float("nan")
    if f1 == 0:
      roots.append(x1)
    if math.isfinite(f1) and abs(f1) < zero_eps:
      roots.append(_newton(f, x1))
    if math.isfinite(f0) and math.isfinite(f1) and _sign(f0) * _sign(f1) < 0:
      root = _bisect(f, x0, x1, f0, f1)
      roots.append(root)
    x0, f0 = x1, f1
  return roots

def find_sign_intervals(expr_ast, xmin=-10.0, xmax=10.0, steps=400, zero_eps=1e-6):
  def f(x):
    return parsing.eval_expr_ast(expr_ast, x)

  intervals = []
  step = (xmax - xmin) / steps
  x0 = xmin
  try:
    f0 = f(x0)
  except (ValueError, ZeroDivisionError):
    f0 = float("nan")

  for i in range(1, steps + 1):
    x1 = xmin + i * step
    try:
      f1 = f(x1)
    except (ValueError, ZeroDivisionError):
      f1 = float("nan")

    if math.isfinite(f0) and math.isfinite(f1):
      if abs(f0) < zero_eps:
        intervals.append(("zero", (x0, x0, True, True)))
      if abs(f1) < zero_eps:
        intervals.append(("zero", (x1, x1, True, True)))
      if _sign(f0) * _sign(f1) < 0:
        root = _bisect(f, x0, x1, f0, f1)
        intervals.append(("zero", (root, root, True, True)))
        if _sign(f0) > 0:
          intervals.append(("pos", (x0, root, False, False)))
        else:
          intervals.append(("neg", (x0, root, False, False)))
        if _sign(f1) > 0:
          intervals.append(("pos", (root, x1, False, False)))
        else:
          intervals.append(("neg", (root, x1, False, False)))
      else:
        # Same sign across interval.
        if _sign(f0) > 0:
          intervals.append(("pos", (x0, x1, False, False)))
        elif _sign(f0) < 0:
          intervals.append(("neg", (x0, x1, False, False)))
    x0, f0 = x1, f1

  # Merge contiguous intervals; leave points as-is.
  merged = []
  for kind, interval in sorted(intervals, key=lambda x: (x[1][0], x[1][1], x[0])):
    if interval[0] == interval[1]:
      merged.append((kind, interval))
      continue
    if not merged:
      merged.append((kind, interval))
      continue
    last_kind, last = merged[-1]
    if last[0] == last[1] or last_kind != kind:
      merged.append((kind, interval))
      continue
    if interval[0] <= last[1] + 1e-9:
      merged[-1] = (kind, (last[0], max(last[1], interval[1]), last[2], interval[3]))
    else:
      merged.append((kind, interval))
  return merged

def intervals_for_op(expr_ast, op, xmin=-10.0, xmax=10.0, steps=400, zero_eps=1e-6):
  tagged = find_sign_intervals(expr_ast, xmin, xmax, steps, zero_eps)
  selected = []
  for kind, interval in tagged:
    if op == "<" and kind == "neg":
      selected.append(interval)
    elif op == "<=" and kind in ("neg", "zero"):
      selected.append(interval)
    elif op == ">" and kind == "pos":
      selected.append(interval)
    elif op == ">=" and kind in ("pos", "zero"):
      selected.append(interval)
  return selected
