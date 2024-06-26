# Capstone design(1)

## 1. Project goal

Optimize **vent hole(outlet) pattern** for **reducing the [cooling drag](https://www.sciencedirect.com/topics/engineering/cooling-drag)** of `UAV`'s motor efficiency.

## 2. Project structure overview

### 2.1 Surrogate modeling
![first pipeline](./assets/1.png)
1. Extract arbitrary outlet data from CFD simulation(ANSYS Fluent)
2. Upscale the input data for high-dimensional surrogate modeling
3. PCA(Principal Component Analysis) for dimension reduction(for train quality)
4. Train the surrogate model with the input data and output data
  - `Input data`: Vent-hole matrix
  - `Output data1`: Drag
  - `Output data2`: Motor surface temperature(avg)
  - `Output data3`: Motor surface temperature(max)

> **Note**: `Drag(Cd)` and `surface temperature(Tavg, mean)` are conflicting objectives. So, we decided to optimize this multi-objective problem using Genetic-algorithm.


### 2.2 Generate vent hole pattern sets
![second pipeline](./assets/2.png)

1. Pattern type selection - defined by transformation group
    - `Grid`
    - `Circular`
    - `Corn`
2. Shape definition - define by closed area function(user-defined) `F(x, y) <= 0`
3. Chromosome(pattern) generation - generate vent hole pattern sets
4. Generate pattern matrix - Model's input data
  Example of generated patterns:
  ![generated patterns](./assets/pattern.png)

### 2.3 Genetic algorithm
![third pipeline](./assets/3.png)

1. Initialize population - can be customized
    - `Shape` gene distribution
    - `Pattern` gene distribution
    - population size, ...etc
2. Evaluate fitness - can be customized
    - `Fitness1`: Drag
    - `Fitness2`: Motor surface temperature(avg)
    - `Fitness3`: Motor surface temperature(max)
3. Selection - can be customized
    - `Tournament selection`
    - `Roulette wheel selection`
    - `Stochastic universal sampling`
    - `Elitism`
4. Crossover - can be customized
    - `Single-point crossover`
    - `Two-point crossover`
    - `Uniform crossover`

![fourth pipeline](./assets/4.png)

5. Mutation - can be customized
    - `Gaussian mutation`(stochastic approach)
    - Random mutation(probabilistic approach)
6. Termination - can be customized  
    - `Max generation`
    - `Population diversity`
    - `Extraordinary fitness value`
    - `...etc`


## 3. Project running

1. Clone this repository
    ```bash
    git clone [repo link]
    ```

2. Install virtual environment

    vscode based(recommended):
    - `ctrl + shift + p` -> `Python: Create environment`.
    - select `./venv` directory.
    - select python interpreter version, `3.11.0` or higher recommended.

3. Install required packages
    ```bash
    pip3 install -r requirements.txt
    ```

4. Run the main script
    1. goto `experimental_template.ipynb` file(make sure you have installed jupyter notebook)
    2. adjust genetic algorithm hyperparameters
    3. run the script

        ```python
        suite = GAPipeline[VentHole](
            suite_name="exp/tournament/config", 
            # 🔼 Name of experiment, and also the directory name store/{suite_name}

            suite_max_count=50, 
            # 🔼 Maximum generation

            suite_min_population=20, 
            # 🔼 Minimum population size

            suite_min_chromosome=40, 
            # 🔼 Minimum unique chromosome size

            crossover_behavior=UniformCrossover(), 
            # 🔼 Crossover behavior

            selector_behavior=TournamentSelectionFilter(tournament_size=4), 
            # 🔼 Selection behavior

            fitness_calculator=VentFitnessCalculator( 
                # 🔼 Fitness model
                model_trainer_tuple=(
                    gpr_model_trainer,
                    gpr_model_trainer,
                    gpr_model_trainer,
                ),
                criteria_weight_list=CRITERIA_WEIGHT,
                drag_criterion=DRAG_CRITERION,
                drag_std_criterion=DRAG_STD_CRITERION,
                avg_temp_criterion=AVG_TEMP_CRITERION,
                avg_temp_std_criterion=AVG_TEMP_STD_CRITERION,
                max_temp_criterion=MAX_TEMP_CRITERION,
                max_temp_std_criterion=MAX_TEMP_STD_CRITERION,
            ),
            immediate_exit_condition=lambda x: x[0] >= 0.725 and x[1] >= 0.725, 
            # 🔼 Exit condition, extra ordinary fitness value

            mutation_probability=0.01,  # 1% 
            # 🔼 Mutation probability

            population_initializer=VentInitializer( 
                population_size=150, 
                # 🔼 Initial population size

                grid_scale=GRID_SCALE,
                grid_resolution=GRID_RESOLUTION,
                pattern_bound=GRID_BOUND,

                pattern_gene_pool=[ 
                    # 🔼 Pattern gene pool, you can adjust the gene pool
                    circular_params,
                    corn_params,
                    grid_params,
                ],

                shape_gene_pool=[ 
                    # 🔼 Shape gene pool, you can adjust the gene pool
                    circle_params,
                ],
            ),
        )
        ```
    4. Check the results
      
        Result example:
        ![result example](./assets/result.png)

      - Gene, fitness data: check `store/{suite_name}` directory
        > **Note**:
        > 
        > All the results are recorded at `store` directory by `json` format data.
        > You can visualize or analyze the results.
