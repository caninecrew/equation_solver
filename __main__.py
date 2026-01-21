from solve import solve

print(solve("piecewise(x<0, -x, x>=0, x, 0) = 1"))


print(solve("1/(x+1) = 2", xmin=-10, xmax=10))
