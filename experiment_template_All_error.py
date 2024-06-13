{
	"name": "IndexError",
	"message": "index 0 is out of bounds for axis 0 with size 0",
	"stack": "---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[2], line 1
----> 1 suite.run()

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\ga_pipeline.py:375, in GAPipeline.run(self)
    373 while not self._terminate_condition:
    374     self._log_generation()
--> 375     self._run_generation()
    376     self._record_generation()
    378 self._post_exit()

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\ga_pipeline.py:323, in GAPipeline._run_generation(self)
    318 self._population = self.selector_behavior.select(
    319     self._population, self.population_storage
    320 )
    322 # Phase 4: Crossover
--> 323 self._crossover_popularization()
    325 # Phase 5: Mutation
    326 self._mutate_popularization()

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\ga_pipeline.py:273, in GAPipeline._crossover_popularization(self)
    268     parent1: ChromosomeType = self._population[random_index]
    269     parent2: ChromosomeType = self._population[
    270         index_list[(i + 1) % self.population_count]
    271     ]
--> 273     children: ChromosomeType = parent1.crossover(
    274         behavior=self.crossover_behavior,
    275         other=parent2,
    276     )
    278     child_population.append(children)
    280 self._population = child_population

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\chromosome\\chromosome.py:122, in Chromosome.crossover(self, behavior, other)
    114 def crossover(
    115     self,
    116     behavior: CrossoverBehavior,
    117     other: \"Chromosome[tuple[Gene, ...]]\",
    118 ) -> \"Chromosome[tuple[Gene, ...]]\":
    119     \"\"\"
    120     Crossover the chromosome with another chromosome
    121     \"\"\"
--> 122     child_chromosome: \"Chromosome[tuple[Gene, ...]]\" = self.crossover_genes(other)
    124     if self.is_gene_parameter_crossover_possible(child_chromosome):
    125         crossover_gene_params = self.crossover_gene_params(
    126             behavior, child_chromosome
    127         )

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\chromosome\\vent_hole.py:41, in VentHole.crossover_genes(self, other_chromosome)
     39 if first_choice == 0:
     40     second_choice = 1
---> 41     child_chromosome = VentHole(
     42         gene_tuple=(
     43             self.gene_tuple[first_choice],
     44             other_chromosome.gene_tuple[second_choice],
     45         ),
     46         vent_id=self.chromosome_id,  # Inherit the ID from the parent
     47         pattern_bound=self.pattern_bound,
     48     )
     49 else:
     50     second_choice = 0

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\chromosome\\vent_hole.py:28, in VentHole.__init__(self, vent_id, gene_tuple, pattern_bound)
     21 super().__init__(
     22     label=\"VentHole Chromosome\",
     23     chromosome_id=vent_id,
     24     gene_tuple=gene_tuple,
     25 )
     27 self.pattern_bound = pattern_bound
---> 28 self.pattern = Pattern(
     29     PatternTransformationMatrix(
     30         self.gene_tuple[0].pattern_unit,
     31         self.gene_tuple[1].param.transformation,
     32         pattern_bound=pattern_bound,
     33     )
     34 )

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\geometry\\pattern_unit.py:577, in Pattern.__init__(self, pattern_transformation_matrix)
    574 self.pattern_transformation_matrix = pattern_transformation_matrix
    576 self.pattern_matrix: V2_group = V.initialize_matrix_2d()
--> 577 self.generate_pattern_matrix()

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\geometry\\pattern_unit.py:604, in Pattern.generate_pattern_matrix(self)
    593 for T_vec in self.pattern_transformation_matrix.T_matrix:
    594     transformed_t_vec = Grid.discretize_points(
    595         Transformer.transform_rt(
    596             self.pattern_unit.shape_matrix,
   (...)
    602         self.pattern_unit.grid.k,
    603     )
--> 604     t_id = f\"{transformed_t_vec[0]}_{transformed_t_vec[1]}\"
    606     # Remove duplicated transformation
    607     if t_id not in t_id_set:

IndexError: index 0 is out of bounds for axis 0 with size 0"
}