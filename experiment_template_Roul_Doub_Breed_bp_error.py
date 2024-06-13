{
	"name": "ZeroDivisionError",
	"message": "float division by zero",
	"stack": "---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
Cell In[2], line 1
----> 1 suite.run()

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\ga_pipeline.py:375, in GAPipeline.run(self)
    373 while not self._terminate_condition:
    374     self._log_generation()
--> 375     self._run_generation()
    376     self._record_generation()
    378 self._post_exit()

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\ga_pipeline.py:326, in GAPipeline._run_generation(self)
    323 self._crossover_popularization()
    325 # Phase 5: Mutation
--> 326 self._mutate_popularization()
    328 # Phase 6: Fitness Calculation for children
    329 if self.check_children_fitness:

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\ga_pipeline.py:286, in GAPipeline._mutate_popularization(self)
    284 random_value = random()
    285 if random_value < self.mutation_probability:
--> 286     chromosome.mutate_genes()
    287     self._mutation_count += 1

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\chromosome\\vent_hole.py:81, in VentHole.mutate_genes(self)
     79 else:
     80     rand_index = choice(range(gene.parameter_count))
---> 81     gene.mutate_at(rand_choice, rand_index)

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\gene\\pattern\\pattern_gene.py:378, in PatternGene.mutate_at(self, method, at)
    375 elif method == \"preserve\":
    376     return
--> 378 self.update_gene(new_parameter_list)

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\gene\\pattern\\pattern_gene.py:236, in PatternGene.update_gene(self, new_parameter_list)
    234 def update_gene(self, new_parameter_list: np.ndarray[np.float64]) -> None:
    235     # 0. Get the fixed transformation
--> 236     new_parameter_list = self._get_fixed_transformation(new_parameter_list)
    238     # Assume that parameter_list is already updated by `mutate` or `mutate_at`
    239     prev_parameter_list: list[float] = self.parameter_list.tolist()

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\gene\\pattern\\pattern_gene.py:188, in PatternGene._get_fixed_transformation(self, new_parameter_list)
    185     elif param_id == \"rot_count\":
    186         ordered_params[4] = new_parameter_list[i]
--> 188 self.param.update_transformation_params(tuple(ordered_params))
    190 fixed_parameter_list = np.zeros(self.parameter_count, dtype=np.float64)
    191 for i, param_id in enumerate(self.param.parameter_id_list):

File c:\\Users\\eaton\\Desktop\\Capstone_Simulation\\capstone_1_er1\\src\\ga\\gene\\pattern\\pattern_gene.py:96, in PatternGeneParameter.update_transformation_params(self, transformation)
     93 if phi == 0:
     94     phi = 0
---> 96 rotation_count = floor(2 * pi / phi)
     98 modified_phi: float = 2 * pi / rotation_count
     99 phi = modified_phi

ZeroDivisionError: float division by zero"
}