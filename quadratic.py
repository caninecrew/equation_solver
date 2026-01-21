def quadratic(a, b, c):
  """
  Calculates the roots of a quadratic equation using the quadratic formula.

  Args:
    a (float): The coefficient of the x^2 term.
    b (float): The coefficient of the x term.
    c (float): The constant term.

  Returns:
    list[float]: A list containing the two roots of the quadratic equation.
  """
  # Calculate the discriminant (b^2 - 4ac)
  discriminant = (b**2) - (4*a*c)
  # Check for real roots
  if discriminant < 0: 
      return [] # No real roots
  sqr = m.sqrt(discriminant) # Calculate the square root of the discriminant
  bottom = 2*a # Calculate the denominator (2a)

  positive = (-b + sqr) / bottom # Calculate the first root using the '+' sign
  negative = (-b - sqr) / bottom # Calculate the second root using the '-' sign

  return [positive, negative] # Return both roots in a list