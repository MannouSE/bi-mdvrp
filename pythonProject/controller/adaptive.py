import l

fitness = [problem.evaluate(sol) for sol in population]
M = mating_pool(population, fitness, pool_size=len(population))
Q_t = apply_variation(M)
