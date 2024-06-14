from math import atan2, cos, pi, sqrt, tan
from src.ga.gene.shape.shape_gene import ShapeGeneParameter


def diamond(p, params) -> bool:
    x, y = p
    a = params[0]
    first = y <= a - abs(x) and y >= -a + abs(x)  # y <= a - |x| and y >= -a + |x|
    second = y >= -a + abs(x) and y <= a - abs(x)  # y >= -a + |x| and y <= a - |x|
    return first and second


def diamond_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="DiamondShape",
        bbox=bbox,
        a_f=[diamond],
        parameter_id_list=["diamond_l"],
        parameter_boundary_list=[(1 * scale, 4 * scale)],
    )


def hexagon(p, params) -> bool:
    x, y = p
    a = params[0]
    s3 = sqrt(3)
    f1 = y >= s3 * (x - a)
    f2 = y <= -s3 * (x - a)
    f4 = y <= s3 * (x + a)
    f3 = y >= -s3 * (x + a)
    f5 = y <= s3 / 2 * a or y >= -s3 / 2 * a

    return f1 and f3 and f5 and f2 and f4


def hexagon_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="HexagonShape",
        bbox=bbox,
        a_f=[hexagon],
        parameter_id_list=["hexagon_l"],
        parameter_boundary_list=[(4 * scale, 6 * scale)],
    )


def parabola_x_right_inner(p, params) -> bool:
    x, y = p
    a, b = params
    return x >= (y**2) / b - a and x <= a


def parabola_x_right_inner_params(
    scale: int, resolution: float = 2.0
) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="ParabolaXRightInnerShape",
        bbox=bbox,
        a_f=[parabola_x_right_inner],
        parameter_id_list=["parabola_xr_inner_a", "parabola_xr_inner_b"],
        parameter_boundary_list=[(2 * scale, 9 * scale), (1 * scale, 5 * scale)],
    )


def parabola_x_right_inner_multi(p, params) -> bool:
    x, y = p
    a1, b1, a2, b2 = params
    return x >= (y**2) / b1 - a1 and x <= y**2 / b2 - a2 and x <= 10


def parabola_x_right_inner_multi_params(
    scale: int, resolution: float = 2.0
) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="ParabolaXRightInnerMultiShape",
        bbox=bbox,
        a_f=[parabola_x_right_inner_multi],
        parameter_id_list=[
            "parabola_xr_inner_a1",
            "parabola_xr_inner_b1",
            "parabola_xr_inner_a2",
            "parabola_xr_inner_b2",
        ],
        parameter_boundary_list=[
            (7 * scale, 9 * scale),
            (1 * scale, 5 * scale),
            (2 * scale, 5 * scale),
            (1 * scale, 3 * scale),
        ],
    )


def arrow(p, params) -> bool:
    x, y = p
    theta1, theta2, arrow_a = params
    return (
        abs(y) <= tan(theta1) * (x + arrow_a)
        and abs(y) >= tan(theta2) * (x)
        and abs(y) >= -arrow_a
    )


def arrow_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="ArrowShape",
        bbox=bbox,
        a_f=[arrow],
        parameter_id_list=["arrow_theta_1", "arrow_theta_2", "arrow_a"],
        parameter_boundary_list=[
            (pi / 20, pi / 6),
            (pi / 5.5, pi / 2.2),
            (2 * scale, 7 * scale),
        ],
    )


def polar_to_cartesian(x, y):
    r = sqrt(x**2 + y**2)
    theta = atan2(y, x)
    return r, theta


def flower(p, params) -> bool:
    x, y = p
    count, radius = params
    count = int(count)
    r, theta = polar_to_cartesian(x, y)
    return r <= radius + cos(count * theta)


def flower_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="FlowerShape",
        bbox=bbox,
        a_f=[flower],
        parameter_id_list=["flower_count", "flower_r"],
        parameter_boundary_list=[(1, 7), (2 * scale, 5 * scale)],
    )


def rose(p, params) -> bool:
    x, y = p
    a, n = params
    n = int(n)
    k = n / 1  # d = 1
    r, theta = polar_to_cartesian(x, y)
    return r <= a * cos(k * theta)


# https://en.wikipedia.org/wiki/Rose_(mathematics)
def rose_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="RoseShape",
        bbox=bbox,
        a_f=[rose],
        parameter_id_list=["rose_a", "rose_n"],
        parameter_boundary_list=[(5 * scale, 15 * scale), (2, 7)],
    )


def trapezoid_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="TrapezoidShape",
        bbox=bbox,
        a_f=[
            lambda p, params: p[1]
            <= ((params[1] - params[0]) / 10 * (p[0] - 5) + params[1])
            and p[1] >= -((params[1] - params[0]) / 10 * (p[0] - 5) + params[1])
            # y <= (k2 - k1) / 10 * (x - 5) + k2 and
        ],
        parameter_id_list=["k1", "k2"],
        parameter_boundary_list=[(2 * scale, 5 * scale), (1 * scale, 5 * scale)],
    )


def triple_rectangle(p, params) -> bool:
    x, y = p
    w, h, delta_y = params
    return (
        (x >= -w and x <= w and y >= -h and y <= h)
        or (x >= -w and x <= w and y >= h + delta_y and y <= 2 * h + delta_y)
        or (x >= -w and x <= w and -1 * y >= h + delta_y and -1 * y <= 2 * h + delta_y)
    )


def triple_rectangle_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="TripleRectangleShape",
        bbox=bbox,
        a_f=[triple_rectangle],
        parameter_id_list=["rect__w", "rect__h", "rect__delta_y"],
        parameter_boundary_list=[
            (2 * scale, 5 * scale),
            (0.75 * scale, 1.5 * scale),
            (0.75 * scale, 2 * scale),
        ],
    )


def circle_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="CircleShape",
        bbox=bbox,
        a_f=[
            lambda p, params: (p[0] ** 2 + p[1] ** 2) <= params[0] ** 2,
            # x^2 + y^2 <= r^2
        ],
        parameter_id_list=["r"],
        parameter_boundary_list=[(1.5 * scale, 3 * scale)],
        # 2 <= r <= 5
    )


def triangle_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="TriangleShape",
        bbox=bbox,
        a_f=[
            lambda p, params: p[0] >= 0 and p[1] >= 0 and p[0] + p[1] <= params[0],
            # x >= 0 and y >= 0 and x + y <= l
        ],
        parameter_id_list=["l"],
        parameter_boundary_list=[(3 * scale, 10 * scale)],
        # 2 <= l <= 10
    )


def wing_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="WingShape",
        bbox=bbox,
        a_f=[
            lambda p, params: p[1] >= p[0] ** 2 - params[0]
            and p[1] <= -p[0] ** 2 + params[0],
            # y <= x^2 - c and y <= -x^2 + c
        ],
        parameter_id_list=["c"],
        parameter_boundary_list=[(2 * scale, 5 * scale)],
    )
