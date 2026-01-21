import cmath

def _eval_poly(coeffs, x):
  value = 0j
  for coef in coeffs:
    value = value * x + coef
  return value

def _durand_kerner(coeffs, max_iter=200, tol=1e-12):
  n = len(coeffs) - 1
  if n <= 0:
    return []

  lead = coeffs[0]
  if lead == 0:
    raise ValueError("Leading coefficient cannot be zero.")

  # Normalize to monic.
  coeffs = [c / lead for c in coeffs]

  roots = [cmath.exp(2j * cmath.pi * k / n) for k in range(n)]
  for _ in range(max_iter):
    converged = True
    for i in range(n):
      prod = 1+0j
      for j in range(n):
        if i == j:
          continue
        diff = roots[i] - roots[j]
        if diff == 0:
          diff = 1e-12 + 1e-12j
        prod *= diff
      delta = _eval_poly(coeffs, roots[i]) / prod
      roots[i] -= delta
      if abs(delta) > tol:
        converged = False
    if converged:
      break
  return roots

def _dedupe_reals(values, eps=1e-7):
  values.sort()
  deduped = []
  for v in values:
    if not deduped or abs(v - deduped[-1]) > eps:
      deduped.append(v)
  return deduped

def roots(coeffs):
  """
  Returns complex roots (decimal approximations) for a polynomial.
  coeffs: list ordered high-to-low (e.g., [a, b, c] for ax^2+bx+c).
  """
  return _durand_kerner(coeffs)

def real_roots(coeffs):
  """
  Returns real roots (decimal approximations) for a polynomial.
  coeffs: list ordered high-to-low (e.g., [a, b, c] for ax^2+bx+c).
  """
  roots = _durand_kerner(coeffs)
  reals = [r.real for r in roots if abs(r.imag) < 1e-7]
  return _dedupe_reals(reals)
