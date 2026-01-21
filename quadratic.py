import cmath

def quadratic(a, b, c):
  """
  Calculates the roots of a quadratic equation using the quadratic formula.

  Args:
    a (float): The coefficient of the x^2 term.
    b (float): The coefficient of the x term.
    c (float): The constant term.

  Returns:
    list[complex]: A list containing the two roots (may be complex).
  """
  discriminant = (b**2) - (4*a*c)
  sqr = cmath.sqrt(discriminant)
  bottom = 2 * a
  positive = (-b + sqr) / bottom
  negative = (-b - sqr) / bottom
  return [positive, negative]
