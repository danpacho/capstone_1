from src.ga.gene.shape.shape_gene import ShapeGene, ShapeParameter

donut_params = ShapeParameter(
    label="DonutShape",
    bbox=(-10, 10, 0.25),
    a_f=[
        lambda p, params: (p[0] ** 2 + p[1] ** 2) >= params[0]
        and (p[0] ** 2 + p[1] ** 2) <= params[1],
        # x^2 + y^2 >= r_inner^2 and x^2 + y^2 <= r_outer^2
    ],
    parameter_id_list=["r_inner", "r_outer"],
    parameter_boundary_list=[(2, 6), (7, 10)],
    # 2 <= r_inner <= 6, 7 <= r_outer <= 10
)

trapezoid_params = ShapeParameter(
    label="TrapezoidShape",
    bbox=(-10, 10, 0.25),
    a_f=[
        lambda p, params: p[0] >= params[0]
        and p[0] <= params[1]
        and p[1] >= params[2]
        and p[1] <= params[3],
        # x >= x1 and x <= x2 and y >= y1 and y <= y2
    ],
    parameter_id_list=["x1", "x2", "y1", "y2"],
    parameter_boundary_list=[(-10, 10), (-10, 10), (-10, 10), (-10, 10)],
    # -10 <= x1, x2, y1, y2 <= 10
)

circle_params = ShapeParameter(
    label="CircleShape",
    bbox=(-12.5, 12.5, 0.25),
    a_f=[
        lambda p, params: (p[0] ** 2 + p[1] ** 2) <= params[0],
        # x^2 + y^2 <= r^2
    ],
    parameter_id_list=["r"],
    parameter_boundary_list=[(2, 5)],
    # 2 <= r <= 5
)

triangle_params = ShapeParameter(
    label="TriangleShape",
    bbox=(-10, 10, 0.25),
    a_f=[
        lambda p, params: p[0] >= 0 and p[1] >= 0 and p[0] + p[1] <= params[0],
        # x >= 0 and y >= 0 and x + y <= l
    ],
    parameter_id_list=["l"],
    parameter_boundary_list=[(2, 10)],
    # 2 <= l <= 10
)

wing_params = ShapeParameter(
    label="WingShape",
    bbox=(-10, 10, 0.25),
    a_f=[
        lambda p, params: p[0] >= 0
        and p[1] >= 0
        and p[1] <= p[0] ** 2 + params[0]
        and p[1] <= -p[0] ** 2 + params[1],
        # x >= 0 and y >= 0 and y <= x^2 + up1 and y <= -x^2 + up2
    ],
    parameter_id_list=["up1", "up2"],
    parameter_boundary_list=[(2, 10), (-10, -2)],
)

DONUT = ShapeGene(shape_parameter=donut_params)
TRAPEZOID = ShapeGene(shape_parameter=trapezoid_params)
CIRCLE = ShapeGene(shape_parameter=circle_params)
TRIANGLE = ShapeGene(shape_parameter=triangle_params)
WING = ShapeGene(shape_parameter=wing_params)
