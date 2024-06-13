from src.ga.gene.shape.shape_gene import ShapeGeneParameter


def donut_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="DonutShape",
        bbox=bbox,
        a_f=[
            lambda p, params: (p[0] ** 2 + p[1] ** 2) >= params[0] ** 2
            and (p[0] ** 2 + p[1] ** 2) <= params[1] ** 2,
            # x^2 + y^2 >= r_inner^2 and x^2 + y^2 <= r_outer^2
        ],
        parameter_id_list=["r_inner", "r_outer"],
        parameter_boundary_list=[(2 * scale, 6 * scale), (7 * scale, 10 * scale)],
        # 2 <= r_inner <= 6, 7 <= r_outer <= 10
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


def hole_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="HoleShape",
        bbox=bbox,
        a_f=[lambda p, params: (p[0] ** 2 + p[1] ** 2) >= params[0]],
        parameter_id_list=["hole_r"],
        parameter_boundary_list=[(2 * scale, 4 * scale)],
    )


def ray_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="RayShape",
        bbox=bbox,
        a_f=[
            lambda p, params: 
            p[0]<=5
            and p[0]>= -5
            and p[1] <= 5
            and p[1]>=-5
            and [1] <= params[0]*p[0]+params[1]+params[4]
            and p[1] <= -params[0]*p[0]+params[1]+params[4]
            and p[1] >= -params[2]* (p[0]-params[3]) ** 2+params[4]
            #and p[1] >= 0
        ],
        parameter_id_list=["upper_slope", "nose_point", "lower_coefficient","lower_x-intercept", "lower_y-intercept"],
        parameter_boundary_list=[(1 * scale, 3 * scale), (1 * scale, 3 * scale), (1 * scale, 4 * scale), (1 * scale, 4 * scale), (1 * scale, 2 * scale)],
    )

def double_parabolic_params(scale: int, resolution: float = 2.0) -> ShapeGeneParameter:
    bbox = (10 * scale, 10 * scale, scale / resolution)
    return ShapeGeneParameter(
        label="DoubleParabolicShape",
        bbox=bbox,
        a_f=[
            lambda p, params: 
            p[0]<=5
            and p[0]>= -5
            and p[1]<=5
            and p[1]>= -5
            and p[1] <= -params[1] * (p[0] ** 2) + params[2] + params[3]
            and p[1] >= -params[0] * ((p[0]+ params[4] ) ** 2) + params[2]
            #and p[1] >= 0
        ],
        parameter_id_list=["lower_coefficient", "upper_coefficient", "lower_y-intercept", "upper_y-intercept_from_lower_y", "lower_x_trans"],
        parameter_boundary_list=[(1 * scale, 3 * scale), (1 * scale, 3 * scale), (1 * scale, 4 * scale), (1 * scale, 2 * scale), (-5 * scale, 5 * scale)],
        # 2 <= r_inner <= 6, 7 <= r_outer <= 10
    )