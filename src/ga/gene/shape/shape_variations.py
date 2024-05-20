from src.ga.gene.shape.shape_gene import ShapeGeneParameter

bbox = (10, 10, 0.25)

donut_params = ShapeGeneParameter(
    label="DonutShape",
    bbox=bbox,
    a_f=[
        lambda p, params: (p[0] ** 2 + p[1] ** 2) >= params[0]
        and (p[0] ** 2 + p[1] ** 2) <= params[1],
        # x^2 + y^2 >= r_inner^2 and x^2 + y^2 <= r_outer^2
    ],
    parameter_id_list=["r_inner", "r_outer"],
    parameter_boundary_list=[(2, 6), (7, 10)],
    # 2 <= r_inner <= 6, 7 <= r_outer <= 10
)

trapezoid_params = ShapeGeneParameter(
    label="Trapezoid",
    bbox=bbox,
    a_f=[
        lambda p, params: p[1]
        <= ((params[1] - params[0]) / 10 * (p[0] - 5) + params[1])
        and p[1] >= -((params[1] - params[0]) / 10 * (p[0] - 5) + params[1])
        # y <= (k2 - k1) / 10 * (x - 5) + k2 and
    ],
    parameter_id_list=["k1", "k2"],
    parameter_boundary_list=[(2, 5), (1, 5)],
)

circle_params = ShapeGeneParameter(
    label="CircleShape",
    bbox=(12.5, 12.5, 0.25),
    a_f=[
        lambda p, params: (p[0] ** 2 + p[1] ** 2) <= params[0],
        # x^2 + y^2 <= r^2
    ],
    parameter_id_list=["r"],
    parameter_boundary_list=[(2, 5)],
    # 2 <= r <= 5
)

triangle_params = ShapeGeneParameter(
    label="TriangleShape",
    bbox=bbox,
    a_f=[
        lambda p, params: p[0] >= 0 and p[1] >= 0 and p[0] + p[1] <= params[0],
        # x >= 0 and y >= 0 and x + y <= l
    ],
    parameter_id_list=["l"],
    parameter_boundary_list=[(3, 10)],
    # 2 <= l <= 10
)

wing_params = ShapeGeneParameter(
    label="WingShape",
    bbox=bbox,
    a_f=[
        lambda p, params: p[1] >= p[0] ** 2 - params[0]
        and p[1] <= -p[0] ** 2 + params[0],
        # y <= x^2 - c and y <= -x^2 + c
    ],
    parameter_id_list=["c"],
    parameter_boundary_list=[(2, 5)],
)

hole_params = ShapeGeneParameter(
    label="HoldShape",
    bbox=bbox,
    a_f=[lambda p, params: (p[0] ** 2 + p[1] ** 2) >= params[0]],
    parameter_id_list=["hole_r"],
    parameter_boundary_list=[(2, 4)],
)
