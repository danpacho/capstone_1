from math import pi

from src.ga.gene.pattern.pattern_gene import PatternGeneParameter

K = pow(2, 20)


def grid_params() -> PatternGeneParameter:
    return PatternGeneParameter(
        pattern_type="grid_strict",
        label="GridStrictTransformation",
        parameter_boundary_list=[
            (0.5 * K, 10.0 * K),  # dx
            (0.5 * K, 10.0 * K),  # dy
        ],
        parameter_id_list=["dx", "dy"],
    )


def circular_params() -> PatternGeneParameter:
    return PatternGeneParameter(
        pattern_type="circular_strict",
        label="CircularStrictTransformation",
        parameter_boundary_list=[
            (7.5 * K, 15 * K),  # di
            (0.5 * K, 10.0 * K),  # dx
            (pi / 10, pi / 4),  # phi,
        ],
        parameter_id_list=["di", "dx", "phi"],
    )


def corn_params() -> PatternGeneParameter:
    return PatternGeneParameter(
        pattern_type="corn",
        label="CornerTransformation",
        parameter_boundary_list=[
            (7.5 * K, 15 * K),  # di
            (0.5 * K, 10.0 * K),  # dx
            (pi / 12, pi / 4),  # phi,
            (2, 7),  # rot_count
        ],
        parameter_id_list=["di", "dx", "phi", "rot_count"],
    )
