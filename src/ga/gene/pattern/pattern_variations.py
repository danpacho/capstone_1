from math import pi

from src.ga.gene.pattern.pattern_gene import PatternGeneParameter


def grid_params(scale: int) -> PatternGeneParameter:
    return PatternGeneParameter(
        pattern_type="grid_strict",
        label="GridStrictTransformation",
        parameter_boundary_list=[
            (0.5 * scale, 10.0 * scale),  # dx
            (0.5 * scale, 10.0 * scale),  # dy
        ],
        parameter_id_list=["dx", "dy"],
    )


def circular_params(scale: int) -> PatternGeneParameter:
    return PatternGeneParameter(
        pattern_type="circular_strict",
        label="CircularStrictTransformation",
        parameter_boundary_list=[
            (7.5 * scale, 15 * scale),  # di
            (0.5 * scale, 10.0 * scale),  # dx
            (pi / 10, pi / 4),  # phi,
        ],
        parameter_id_list=["di", "dx", "phi"],
    )


def corn_params(scale: int) -> PatternGeneParameter:
    return PatternGeneParameter(
        pattern_type="corn",
        label="CornTransformation",
        parameter_boundary_list=[
            (7.5 * scale, 15 * scale),  # di
            (0.5 * scale, 10.0 * scale),  # dx
            (pi / 12, pi / 4),  # phi,
            (2, 7),  # rot_count
        ],
        parameter_id_list=["di", "dx", "phi", "rot_count"],
    )
