import math as m
from fractions import Fraction

def _simplify_sqrt(n):
  if n < 0:
    raise ValueError("Negative discriminant.")
  root = int(m.isqrt(n))
  if root * root == n:
    return root, 1
  factor = 1
  for k in range(2, int(m.sqrt(n)) + 1):
    sq = k * k
    if n % sq == 0:
      factor = k
      n //= sq
  return factor, n

def _format_fraction(value):
  if value.denominator == 1:
    return str(value.numerator)
  return f"{value.numerator}/{value.denominator}"

def _format_root(a, b, disc, sign):
  den = 2 * a
  num = -b
  coef, radicand = _simplify_sqrt(disc)

  rational = Fraction(num, den)
  sqrt_coef = Fraction(coef, den)
  if sign < 0:
    sqrt_coef = -sqrt_coef

  parts = []
  if rational != 0:
    parts.append(_format_fraction(rational))

  if radicand != 1:
    if sqrt_coef == 1:
      sqrt_part = f"√{radicand}"
    elif sqrt_coef == -1:
      sqrt_part = f"-√{radicand}"
    else:
      sqrt_part = f"{_format_fraction(sqrt_coef)}√{radicand}"
    parts.append(sqrt_part)

  if not parts:
    exact = "0"
  else:
    exact = " + ".join(parts).replace("+ -", "- ")

  approx = (-b + sign * m.sqrt(disc)) / (2 * a)
  return f"{exact} ≈ {approx}"

def quadratic(a, b, c):
  """
  Calculates the roots of a quadratic equation using the quadratic formula.

  Args:
    a (float): The coefficient of the x^2 term.
    b (float): The coefficient of the x term.
    c (float): The constant term.

  Returns:
    list[str]: A list containing the two roots in exact+decimal form.
  """
  discriminant = (b**2) - (4*a*c)
  if discriminant < 0:
      return []

  return [
    _format_root(a, b, discriminant, 1),
    _format_root(a, b, discriminant, -1),
  ]
