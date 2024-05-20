from math import pi

from src.ga.gene.pattern.pattern_gene import PatternGeneParameter

grid_params = PatternGeneParameter(
    pattern_type="grid_strict",
    label="GridStrictTransformation",
    parameter_boundary_list=[
        (0.5, 10.0),  # dx
        (0.5, 10.0),  # dy
    ],
    parameter_id_list=["dx", "dy"],
)

circular_params = PatternGeneParameter(
    pattern_type="circular_strict",
    label="CircularStrictTransformation",
    parameter_boundary_list=[
        (7.5, 15),  # di
        (0.5, 10.0),  # dx
        (pi / 12, pi / 2),  # phi,
    ],
    parameter_id_list=["di", "dx", "phi"],
)

corn_params = PatternGeneParameter(
    pattern_type="corn",
    label="CornerTransformation",
    parameter_boundary_list=[
        (7.5, 15),  # di
        (0.5, 10.0),  # dx
        (pi / 12, pi / 2),  # phi,
        (1, 7),  # rot_count
    ],
    parameter_id_list=["di", "dx", "phi", "rot_count"],
)
