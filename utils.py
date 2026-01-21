def fix_zero(value, eps=1e-12):
    return 0.0 if abs(value) < eps else value