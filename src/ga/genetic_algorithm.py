from src.prediction.gpr_model_trainer import GPRModelTrainer
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
from src.ga.p2_fitness.vent_fitness import Criterion, VentFitnessCalculator
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

# ----------------- Define the GA CONSTANTS -----------------
# 1. Define the criteria with direction, min, and max values
CRITERIA_WEIGHT = (
    1.2,
    1,
    0.8,
)
# Define the criteria with direction, min, and max values
DRAG_CRITERION: Criterion = ("lower", 0.2, 0.3)  # Lower is better, range 0.2 to 0.5
DRAG_STD_CRITERION: Criterion = ("lower", 0, 0.05)  # Lower is better, range 0 to 0.05
AVG_TEMP_CRITERION: Criterion = ("lower", 250, 400)  # Lower is better, range 250 to 400
AVG_TEMP_STD_CRITERION: Criterion = ("lower", 0, 10)  # Lower is better, range 0 to 50
MAX_TEMP_CRITERION: Criterion = ("lower", 300, 500)  # Lower is better, range 300 to 500
MAX_TEMP_STD_CRITERION: Criterion = (
    "lower",
    0,
    10,
)  # Lower is better, range 300 to 500

# 2. Define the grid parameters
GRID_SCALE = 1
GRID_RESOLUTION = 2.0
GRID_WIDTH = 60
GRID_BOUND = (
    (-GRID_WIDTH / 2, GRID_WIDTH / 2),
    (-GRID_WIDTH / 2, GRID_WIDTH / 2),
)

# ----------------- Define the GA MODEL     -----------------

model_trainer = GPRModelTrainer(
    vent_bound=GRID_BOUND,
    vent_bound_width=GRID_WIDTH,
    vent_resolution=GRID_RESOLUTION,
    vent_scale=GRID_SCALE,
    desired_variance=0.9,
)

gpr_models = model_trainer.get_trained_gpr_models()
pca = model_trainer.get_pca()

# ----------------- Define the GA PIPELINES -----------------

suite1 = GAPipeline[VentHole](
    suite_name="suite_1",
    suite_max_count=50,
    suite_min_population=200,
    crossover_behavior=OnePointCrossover(),
    selector_behavior=TournamentSelectionFilter(tournament_size=15),
    fitness_calculator=VentFitnessCalculator(
        gpr_models=gpr_models,
        pca=pca,
        criteria_weight_list=CRITERIA_WEIGHT,
        drag_criterion=DRAG_CRITERION,
        drag_std_criterion=DRAG_STD_CRITERION,
        avg_temp_criterion=AVG_TEMP_CRITERION,
        avg_temp_std_criterion=AVG_TEMP_STD_CRITERION,
        max_temp_criterion=MAX_TEMP_CRITERION,
        max_temp_std_criterion=MAX_TEMP_STD_CRITERION,
    ),
    immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000,
    mutation_probability=0.01,
    population_initializer=VentInitializer(
        population_size=1000,
        grid_scale=GRID_SCALE,
        grid_resolution=GRID_RESOLUTION,
        pattern_bound=GRID_BOUND,
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

suite2 = GAPipeline[VentHole](
    suite_name="suite_2",
    suite_max_count=50,
    suite_min_population=500,
    crossover_behavior=TwoPointCrossover(),
    selector_behavior=ElitismSelectionFilter(elitism_criterion=0.75),
    fitness_calculator=VentFitnessCalculator(
        gpr_models=gpr_models,
        pca=pca,
        criteria_weight_list=CRITERIA_WEIGHT,
        drag_criterion=DRAG_CRITERION,
        drag_std_criterion=DRAG_STD_CRITERION,
        avg_temp_criterion=AVG_TEMP_CRITERION,
        avg_temp_std_criterion=AVG_TEMP_STD_CRITERION,
        max_temp_criterion=MAX_TEMP_CRITERION,
        max_temp_std_criterion=MAX_TEMP_STD_CRITERION,
    ),
    immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000,
    mutation_probability=0.01,
    population_initializer=VentInitializer(
        population_size=2000,
        grid_scale=GRID_SCALE,
        grid_resolution=GRID_RESOLUTION,
        pattern_bound=GRID_BOUND,
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
    suite_min_population=200,
    crossover_behavior=UniformCrossover(),
    selector_behavior=RouletteWheelSelectionFilter(roulette_pointer_count=4),
    fitness_calculator=VentFitnessCalculator(
        gpr_models=gpr_models,
        pca=pca,
        criteria_weight_list=CRITERIA_WEIGHT,
        drag_criterion=DRAG_CRITERION,
        drag_std_criterion=DRAG_STD_CRITERION,
        avg_temp_criterion=AVG_TEMP_CRITERION,
        avg_temp_std_criterion=AVG_TEMP_STD_CRITERION,
        max_temp_criterion=MAX_TEMP_CRITERION,
        max_temp_std_criterion=MAX_TEMP_STD_CRITERION,
    ),
    immediate_exit_condition=lambda x: x[0] >= 10000 and x[1] >= 10000,
    mutation_probability=0.01,
    population_initializer=VentInitializer(
        population_size=2000,
        grid_scale=GRID_SCALE,
        grid_resolution=GRID_RESOLUTION,
        pattern_bound=GRID_BOUND,
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
