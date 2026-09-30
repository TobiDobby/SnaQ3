"""The sole conflict payoff definition."""

PAYOFF = (
    ((1.0, .5), (0.0, 0.0), (.75, .75), (0.0, 0.0)),
    ((0.0, 0.0), (.75, .75), (0.0, 0.0), (1.0, .5)),
    ((0.0, 0.0), (.75, .75), (0.0, 0.0), (.5, 1.0)),
    ((.5, 1.0), (0.0, 0.0), (.75, .75), (0.0, 0.0)),
)


def target(x_a: int, x_b: int) -> int:
    if x_a not in range(4) or x_b not in range(4):
        raise ValueError("inputs must be in Z_4")
    return (x_a & x_b) & 1  # low (second written) bit of the two-bit AND


def wins(x_a: int, x_b: int, a: int, b: int) -> bool:
    return (a ^ b) == target(x_a, x_b)


def utility(x_a: int, x_b: int, a: int, b: int) -> tuple[float, float]:
    target(x_a, x_b)
    if a not in (0, 1) or b not in (0, 1):
        raise ValueError("outputs must be binary")
    return PAYOFF[2 * a + b][x_a & x_b]


def movement_policy(x_a: int, x_b: int, payoff: tuple[float, float]) -> tuple[int | None, int | None]:
    if payoff == (1.0, .5):
        return x_a, x_a
    if payoff == (.5, 1.0):
        return x_b, x_b
    if payoff == (.75, .75):
        return x_a, x_b
    if payoff == (0.0, 0.0):
        return None, None
    raise ValueError("payoff is not in the canonical table")
