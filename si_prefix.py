import math


def sign(x, value=1):
    """Mathematical signum function.

    :param x: Object of investigation
    :param value: The size of the signum (defaults to 1)
    :returns: Plus or minus value
    """
    return -value if x < 0 else value


def prefix(x, dimension=1):
    """Give the number an appropriate SI prefix.

    :param x: Too big or too small number.
    :returns: String containing a number between 1 and 1000 and SI prefix.
    """
    if x == 0:
        return "0  "

    floor_log10 = math.floor(math.log10(abs(x)))
    if abs(floor_log10) > 24:
        floor_log10 = sign(floor_log10, value=24)

    div, mod = divmod(floor_log10, 3 * dimension)
    return "%.3g %s" % (x * 10**(-floor_log10 + mod), " kMGTPEZYyzafpnμm"[div])
