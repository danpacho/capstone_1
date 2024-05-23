from src.ga.chromosome.vent_hole import VentHole

from src.ga.gene.shape.shape_variations import (
    circle_params,
    donut_params,
    hole_params,
    trapezoid_params,
    triangle_params,
    wing_params,
)
from src.ga.gene.pattern.pattern_variations import (
    circular_params,
    corn_params,
    grid_params,
)

from src.ga.ga_pipeline import GAPipeline

from src.ga.p1_initialize.init_vent import VentInitializer
from src.ga.p2_fitness.vent_fitness import VentFitnessCalculator
from src.ga.p3_select.behaviors import (
    TournamentSelectionFilter,
    ElitismSelectionFilter,
    RouletteWheelSelectionFilter,
)
from src.ga.p4_crossover.behaviors import (
    OnePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
)


criteria_weight_list = (
    1.2,
    0.8,
    1,
)  # Define the criteria with direction, min, and max values
drag_criterion = ("lower", 0.2, 0.5)  # Lower is better, range 0.2 to 0.5
max_temp_criterion = ("lower", 300, 500)  # Lower is better, range 300 to 500
avg_temp_criterion = ("higher", 250, 400)  # Higher is better, range 250 to 400


suite1 = GAPipeline[VentHole](
    suite_name="suite_1",
    suite_max_count=50,
    suite_min_population=50,
    crossover_behavior=OnePointCrossover(),
    fitness_calculator=VentFitnessCalculator(
        criteria_weight_list=criteria_weight_list,
        drag_criterion=drag_criterion,
        max_temp_criterion=max_temp_criterion,
        avg_temp_criterion=avg_temp_criterion,
    ),
    immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000,
    mutation_probability=0.001,
    population_initializer=VentInitializer(
        grid_scale=10**20,
        grid_resolution=2.0,
        pattern_bound=((-30, 30), (-30, 30)),
        population_size=100,
        pattern_gene_pool=[circular_params, corn_params, grid_params],
        shape_gene_pool=[
            circle_params,
            donut_params,
            wing_params,
            hole_params,
            trapezoid_params,
            triangle_params,
        ],
    ),
    selector_behavior=TournamentSelectionFilter(tournament_size=5),
)

suite2 = GAPipeline[VentHole](
    suite_name="suite_2",
    suite_max_count=50,
    suite_min_population=50,
    crossover_behavior=TwoPointCrossover(),
    selector_behavior=ElitismSelectionFilter(elitism_criterion=0.5),
    fitness_calculator=VentFitnessCalculator(
        criteria_weight_list=criteria_weight_list,
        drag_criterion=drag_criterion,
        max_temp_criterion=max_temp_criterion,
        avg_temp_criterion=avg_temp_criterion,
    ),
    immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000,
    mutation_probability=0.001,
    population_initializer=VentInitializer(
        grid_scale=10**20,
        grid_resolution=2.0,
        pattern_bound=((-30, 30), (-30, 30)),
        population_size=100,
        pattern_gene_pool=[circular_params, corn_params, grid_params],
        shape_gene_pool=[
            circle_params,
            donut_params,
            wing_params,
            hole_params,
            trapezoid_params,
            triangle_params,
        ],
    ),
)

suite3 = GAPipeline[VentHole](
    suite_name="suite_3",
    suite_max_count=50,
    suite_min_population=50,
    crossover_behavior=UniformCrossover(),
    selector_behavior=RouletteWheelSelectionFilter(roulette_pointer_count=4),
    fitness_calculator=VentFitnessCalculator(
        criteria_weight_list=criteria_weight_list,
        drag_criterion=drag_criterion,
        max_temp_criterion=max_temp_criterion,
        avg_temp_criterion=avg_temp_criterion,
    ),
    immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000,
    mutation_probability=0.001,
    population_initializer=VentInitializer(
        grid_scale=10**20,
        grid_resolution=2.0,
        pattern_bound=((-30, 30), (-30, 30)),
        population_size=100,
        pattern_gene_pool=[circular_params, corn_params, grid_params],
        shape_gene_pool=[
            circle_params,
            donut_params,
            wing_params,
            hole_params,
            trapezoid_params,
            triangle_params,
        ],
    ),
)
