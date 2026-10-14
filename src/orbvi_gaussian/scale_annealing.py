"""Optional training scale bound in metres; no renderer or loss changes."""
import math


def validate_scale_annealing(target_m, initial_m=None, start=0, end=0, iterations=None):
    if initial_m is None:
        if start != 0 or end != 0:
            raise ValueError('Scale annealing steps require gaussian_initial_max_scale_m')
        return  # Preserve legacy model configurations, including disabled cap=0.
    if not (math.isfinite(initial_m) and math.isfinite(target_m)
            and initial_m >= target_m > 0):
        raise ValueError('Scale annealing requires finite initial >= target > 0 metres')
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('Scale annealing requires integer 0 <= start < end')
    if iterations is not None and end >= iterations:
        raise ValueError('Scale annealing end must be less than training iterations')


def gaussian_scale_cap_m(step, target_m, initial_m=None, start=0, end=0):
    """Log-linear interpolation with exact constant values at both endpoints."""
    validate_scale_annealing(target_m, initial_m, start, end)
    if initial_m is None:
        return target_m
    if step >= end or initial_m == target_m:
        return target_m
    if step <= start:
        return initial_m
    fraction = (step - start) / (end - start)
    return math.exp(math.log(initial_m) + fraction * (math.log(target_m) - math.log(initial_m)))
