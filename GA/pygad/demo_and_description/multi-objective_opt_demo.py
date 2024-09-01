import pygad
import numpy

"""
Given these 2 functions:
    y1 = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    y2 = f(w1:w6) = w1x7 + w2x8 + w3x9 + w4x10 + w5x11 + 6wx12
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=50
    and   (x7,x8,x9,x10,x11,x12)=(-2,0.7,-9,1.4,3,5) and y=30
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize these 2 functions.
This is a multi-objective optimization problem.

PyGAD considers the problem as multi-objective if the fitness function returns:
    1) List.
    2) Or tuple.
    3) Or numpy.ndarray.
"""

function_inputs1 = [4,-2,3.5,5,-11,-4.7] # Function 1 inputs.
function_inputs2 = [-2,0.7,-9,1.4,3,5] # Function 2 inputs.
desired_output1 = 50 # Function 1 output.
desired_output2 = 30 # Function 2 output.

def fitness_func(ga_instance, solution, solution_idx):
    output1 = numpy.sum(solution*function_inputs1)
    output2 = numpy.sum(solution*function_inputs2)
    fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
    fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
    return [fitness1, fitness2]

num_generations = 100
num_parents_mating = 10

sol_per_pop = 20
num_genes = len(function_inputs1)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       parent_selection_type='nsga2')

ga_instance.run()

ga_instance.plot_fitness(label=['Obj 1', 'Obj 2'])

'''
这里的best_solution方法是通过查询适应度最大值的索引, 从而得到最优解和最优适应度值的, 就单目标优化问题而言, 这种方法没有问题, 
但对于多目标优化问题, 多目标优化的核心在于同时优化多个相互竞争的目标，这种方法并不合适。

但是! ga_instance的parent_selection_type为nsga2(非支配排序遗传算法).

疑惑: 一起使用nsga2作为父代选择函数和best_solution这种在计算完所有代的适应度值后再提取最优解的方法合适吗?疑惑主要来自以往的直觉,
以往总觉得最优解总是在动态的变化, 而不是像best_solution这样直接提取.
'''
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")

prediction = numpy.sum(numpy.array(function_inputs1)*solution)
print(f"Predicted output 1 based on the best solution : {prediction}")
prediction = numpy.sum(numpy.array(function_inputs2)*solution)
print(f"Predicted output 2 based on the best solution : {prediction}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
    