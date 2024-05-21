from src.ga.gene.shape.shape_gene import ShapeGeneParameter

# GRID width size
K = pow(2, 20)
# Bounding box, (width, height, resolution)
bbox = (10 * K, 10 * K, K / 2)


def donut_params() -> ShapeGeneParameter:
    return ShapeGeneParameter(
        label="DonutShape",
        bbox=bbox,
        a_f=[
            lambda p, params: (p[0] ** 2 + p[1] ** 2) >= params[0] ** 2
            and (p[0] ** 2 + p[1] ** 2) <= params[1] ** 2,
            # x^2 + y^2 >= r_inner^2 and x^2 + y^2 <= r_outer^2
        ],
        parameter_id_list=["r_inner", "r_outer"],
        parameter_boundary_list=[(2 * K, 6 * K), (7 * K, 10 * K)],
        # 2 <= r_inner <= 6, 7 <= r_outer <= 10
    )


def trapezoid_params() -> ShapeGeneParameter:
    return ShapeGeneParameter(
        label="Trapezoid",
        bbox=(10 * K, 5 * K, K / 2),
        a_f=[
            lambda p, params: p[1]
            <= ((params[1] - params[0]) / 10 * (p[0] - 5) + params[1])
            and p[1] >= -((params[1] - params[0]) / 10 * (p[0] - 5) + params[1])
            # y <= (k2 - k1) / 10 * (x - 5) + k2 and
        ],
        parameter_id_list=["k1", "k2"],
        parameter_boundary_list=[(2 * K, 5 * K), (1 * K, 5 * K)],
    )


def circle_params() -> ShapeGeneParameter:
    return ShapeGeneParameter(
        label="CircleShape",
        bbox=bbox,
        a_f=[
            lambda p, params: (p[0] ** 2 + p[1] ** 2) <= params[0] ** 2,
            # x^2 + y^2 <= r^2
        ],
        parameter_id_list=["r"],
        parameter_boundary_list=[(1.5 * K, 3 * K)],
        # 2 <= r <= 5
    )


def triangle_params() -> ShapeGeneParameter:
    return ShapeGeneParameter(
        label="TriangleShape",
        bbox=bbox,
        a_f=[
            lambda p, params: p[0] >= 0 and p[1] >= 0 and p[0] + p[1] <= params[0],
            # x >= 0 and y >= 0 and x + y <= l
        ],
        parameter_id_list=["l"],
        parameter_boundary_list=[(3 * K, 10 * K)],
        # 2 <= l <= 10
    )


def wing_params() -> ShapeGeneParameter:
    return ShapeGeneParameter(
        label="WingShape",
        bbox=bbox,
        a_f=[
            lambda p, params: p[1] >= p[0] ** 2 - params[0]
            and p[1] <= -p[0] ** 2 + params[0],
            # y <= x^2 - c and y <= -x^2 + c
        ],
        parameter_id_list=["c"],
        parameter_boundary_list=[(2 * K, 5 * K)],
    )


def hole_params() -> ShapeGeneParameter:
    return ShapeGeneParameter(
        label="HoldShape",
        bbox=bbox,
        a_f=[lambda p, params: (p[0] ** 2 + p[1] ** 2) >= params[0]],
        parameter_id_list=["hole_r"],
        parameter_boundary_list=[(2 * K, 4 * K)],
    )
