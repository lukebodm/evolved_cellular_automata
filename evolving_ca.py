import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import random
import pickle
import os
import time

####################################################################################

class CA:
    
    def __init__(self, simulation):
        
        self.rule_dna = np.random.randint(0, 2, size=512)
        self.grid = simulation.solution_grids[0]
        self.solution_grids = simulation.solution_grids
        self.all_possible_rules = simulation.all_possible_rules
        self.number_of_updates = simulation.number_of_updates
        self.fitness_score = 0
        self.t = 0


    def update(self):
        """
        update the cellular automata one time step
        """
        # create a new grid to store the updated values
        new_grid = np.zeros(self.grid.shape)
        # Iterate over each cell in the grid
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # Get the neighbors for the current cell
                neighbors = self.get_neighbors(i, j)

                # Find the index in the rule_dna list corresponding to the neighbor configuration
                index = self.find_rule(neighbors)

                # Update the current cell based on the value at that index in the rule_dna list
                new_grid[i, j] = self.rule_dna[index]

        # Update the grid attribute of the ca object with the new grid
        self.grid = new_grid

        # increment time
        self.t += 1

        # update score
        self.fitness()

    def get_neighbors(self, i, j):
        """
        find values of neighbors in a moore nieghborhood
        """
        # define a list of relative indices for each of the 8 neighbors in a Moore neighborhood
        neighbors_idx = [ (-1, -1), (-1, 0), (-1, 1),
                          ( 0, -1), ( 0, 0), ( 0, 1),
                          ( 1, -1), ( 1, 0), ( 1, 1)]

        neighbors = []
        # Iterate over the relative indices to find the actual indices of the neighbors
        for di, dj in neighbors_idx:
            ni, nj = i+di, j+dj
            # Check if the neighbor is within the bounds of the grid, 
            # and append its value to the neighbor's list
            if 0 <= ni < self.grid.shape[0] and 0 <= nj < self.grid.shape[1]:
                neighbors.append(int(self.grid[ni, nj]))
            else:
                neighbors.append(int(0))

        return neighbors

    def find_rule(self, neighbors):
        """
        input:   a list of neighbor values
        output:  an index to check ones own rule list for update value
        
        this function might be really slow. I need to time it.
        """
        # use numpy's all function to check for equality between each 
        # row of neighbors and all__posible_rules
        match_rows = np.all(self.all_possible_rules == neighbors, axis=1)

        # use numpy's where function to find the indices where x is found in arg
        index = np.where(match_rows)

        # print the indices
        return index
    
    def fitness(self):
        """
        calculate fitness score by comparing to solution grid at same t
        """
        solution_grid = self.solution_grids[self.t]

        # Count the number of cells that differ between the two grids
        diff_count = np.count_nonzero(self.grid != solution_grid)

        # Normalize the score by the total number of cells
        score = 1 - diff_count / self.grid.size

        self.fitness_score += score
        
    def plot(self):
        """
        plot 2D cellular automata at current t
        """
        plt.imshow(self.grid, cmap='binary')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.show()

        
############################################################################

class Simulation:
    
    def __init__(self, solution_grids, pop_size, target_fitness, elite_percentage, reproducers_percentage, mutation_rate, CA_list=False):
        
        self.solution_grids = solution_grids
        self.pop_size = pop_size
        self.grid = solution_grids[0]           # make the initial state match the first state of the solution.
        self.target_fitness = target_fitness    # number between 0 and 1, 1 being perfection (loop never stops) 
        self.number_of_updates = len(solution_grids)-1
        self.elite_number = int(pop_size*elite_percentage)
        self.reproducers_percentage = reproducers_percentage
        self.mutation_rate = mutation_rate
        self.rule_size = 512
        self.all_possible_rules = np.array(list(product([0, 1], repeat=9)))
        
        # if user passed in a CA list, use it, if not, make a fresh one
        if CA_list == False:
            self.CAs = [CA(self) for object in range(pop_size)]
        else:
            self.CAs = CA_list
        
    def reset_CAs(self):
        for ca in self.CAs:
            ca.t = 0
            ca.fitness_score = 0
            ca.grid = self.solution_grids[0]
            
        
    def two_point_crossover(self, ca1, ca2):
        # Select two random points for crossover
        while True:
            point1 = random.randint(0, len(ca1.rule_dna) - 1)
            point2 = random.randint(0, len(ca1.rule_dna) - 1)
            if abs(point1 - point2) >= 45:
                break
            else:
                continue
        if point2 < point1:
            point1, point2 = point2, point1

        # Perform crossover on the rule_dna attributes
        ca1_rule_dna = ca1.rule_dna
        ca2_rule_dna = ca2.rule_dna
        child1 = CA(self)
        child2 = CA(self)
        child1.rule_dna = np.concatenate([ca1_rule_dna[:point1], ca2_rule_dna[point1:point2], ca1_rule_dna[point2:]])
        child2.rule_dna = np.concatenate([ca2_rule_dna[:point1], ca1_rule_dna[point1:point2], ca2_rule_dna[point2:]])

        return child1, child2


    def mutate(self, offspring):
        
        # calculate the number of entries to mutate
        num_mutations = int(self.rule_size * self.mutation_rate)
        
        for ca in offspring:
            # randomly select indices to mutate
            indices = np.random.choice(self.rule_size, num_mutations, replace=False)

            # randomly set selected indices to 0 or 1
            ca.rule_dna[indices] = np.random.randint(2, size=num_mutations)

            
    def update_generation(self):
        for ca in self.CAs:
            for _ in range(self.number_of_updates):
                ca.update()
    
    
    def compute_next_gen(self):
        """ 
        input:  a list of cellular automata
        output: a separate list that represents the next_generation
        """
    
        # sort the CAs based on fitness_score
        sorted_cas = sorted(self.CAs, key=lambda x: x.fitness_score, reverse=True)

        # select the elite and the breeding population
        elite = sorted_cas[:self.elite_number]
        top_percent = sorted_cas[:int(self.pop_size*self.reproducers_percentage)]

        # calculate the number of offspring to produce
        num_offspring = self.pop_size - self.elite_number

        # create a list to store the offspring
        offspring = []

        # loop through the range of offspring to produce
        for i in range(num_offspring//2):
            # randomly select two parents from the top list
            parent1, parent2 = random.choices(top_percent, weights=[x.fitness_score for x in top_percent], k=2)

            # perform two-point crossover between the parents
            child1, child2 = self.two_point_crossover(parent1, parent2)

            # add to offspring object
            offspring.extend([child1, child2])

        # reset the elite fitness_score, time, and initial_state
        for e in elite:
            e.fitness_score = 0
            e.t = 0
            e.grid = self.solution_grids[0]

        # mutate offspring
        self.mutate(offspring)

        # combine the top 10% and the offspring into a new list
        next_generation = elite + offspring
    
        return next_generation
    
    
    def run(self):
        
        # Create a folder with a timestamp as its name to save each generation
        folder_name = time.strftime('saved_sims/%Y-%m-%d_%H-%M-%S')
        os.makedirs(folder_name)
        
        highest_fitness_score = 0
        iteration = 0
        
        begining_time = time.monotonic()
        
        # Run simulations
        while(iteration < 1000):
            start_time = time.monotonic()
            
            self.update_generation()

            # find highest_fitness_score
            sorted_cas = sorted(self.CAs, key=lambda x: x.fitness_score, reverse=True)
            highest_fitness_score = sorted_cas[0].fitness_score
            combined_scores = [x.fitness_score for x in sorted_cas]
            
            # log some information
            print("generation number ", iteration)
            print("highest fitness score: ", round(highest_fitness_score, 7), "     \
                   total fitness score: ", round(sum(combined_scores), 7))
            
            # open a file for writing
            file_name = f"{folder_name}/ca_generation_{iteration}.pkl"
            with open(file_name, 'wb') as f:
                # Write the simulation object to the file
                pickle.dump(self, f)

            if highest_fitness_score >= self.target_fitness:
                break

            # build the next generation
            next_gen = self.compute_next_gen()
            
            end_time = time.monotonic()
            generation_time = end_time - start_time
            
            print(f"runtime: {generation_time:.2f} seconds\n")
            
            self.CAs = next_gen

            # print the generation count
            iteration += 1
            
        final_time = time.monotonic()
        total_time = final_time - begining_time 
        print(f"\ntotal runtime: {total_time:.2f} seconds\n")
        return
    
############################################################################
    
def plot_ca(CA):
    """
    plot any cellular automata
    """
    copy_ca = CA
    copy_ca.grid = copy_ca.solution_grids[0]
    copy_ca.t = 0
    
    # create a list to hold the grids for each time step
    grids = []
    grids.append(copy_ca.solution_grids[0])
    for i in range(copy_ca.number_of_updates):
        copy_ca.update()
        grids.append(copy_ca.grid)

    # prepare a voxels element for matplotlib
    # basically a 3D numpy array for the grids at each t
    voxels = np.stack([grid for grid in grids])

    # plot
    ax = plt.figure().add_subplot(projection='3d')
    if CA.number_of_updates < 7:
        ax.set_xlim3d(0, 16)
    else:
        ax.set_xlim3d(0, CA.number_of_updates)
    ax.set_ylim3d(0, CA.grid.shape[0])
    ax.set_zlim3d(0, CA.grid.shape[1])
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('$t$', fontsize=20)
    ax.xaxis.set_pane_color((0, 0, 0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_facecolor('white') 
    ax.xaxis._axinfo["grid"]['color'] =  (.61,.81,.87,1)
    ax.yaxis._axinfo["grid"]['color'] =  (.61,.81,.87,1)
    ax.zaxis._axinfo["grid"]['color'] =  (.61,.81,.87,1)
    ax.view_init(-45, -45, 20)
    
    plt.show()

def plot_solution_grid(solution_grids):
    """
    plot any solution grid
    """
    # create a list to hold the grids for each time step
    grids = []

    for i in range(len(solution_grids)):
        grids.append(solution_grids[i])

    # prepare a voxels element for matplotlib
    voxels = np.stack([grid for grid in grids])

    # plot
    ax = plt.figure().add_subplot(projection='3d')

    if len(solution_grids) < 7:
        ax.set_xlim3d(0, 16)
    else:
        ax.set_xlim3d(0, len(solution_grids))
    #ax.set_xlim3d(0, len(solution_grids))
    ax.set_ylim3d(0, solution_grids[0].shape[0])
    ax.set_zlim3d(0, solution_grids[0].shape[1])
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('$t$', fontsize=20)
    ax.xaxis.set_pane_color((0, 0, 0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_facecolor('white') 
    ax.xaxis._axinfo["grid"]['color'] =  (.61,.81,.87,1)
    ax.yaxis._axinfo["grid"]['color'] =  (.61,.81,.87,1)
    ax.zaxis._axinfo["grid"]['color'] =  (.61,.81,.87,1)
    ax.view_init(-45, -45, 20)
    
    plt.show()
    
    
def get_centered_sq(size, center_sq_size):
    """ 
    creates a square of 1's in the center of the grid for an initial state
    """
    
    center_sq = np.zeros((size, size))
    center_x, center_y = center_sq.shape[0] // 2, center_sq.shape[1] // 2
    half_sq_size = int(center_sq_size / 2)
    center_sq[center_x-half_sq_size:center_x+half_sq_size, center_y-half_sq_size:center_y+half_sq_size] = 1
    
    return center_sq

def pyramid_solution_grids():
    pyramid_solution_grids = []
    
    grid_size = 16
    initial_sq_size = 10
    
    for i in range(5):
        pyramid_solution_grids.append(get_centered_sq(grid_size, initial_sq_size-(2*i)))
    
    return pyramid_solution_grids

def nail_solution_grids():
    
    nail_solution_grids = []
    
    grid_size = 22
    initial_sq_size = 15
    
    for i in range(27):
        if i <= 3:
            nail_solution_grids.append(get_centered_sq(grid_size, initial_sq_size-(2*i)))
        elif 3<i<=22:
            nail_solution_grids.append(get_centered_sq(grid_size, initial_sq_size-(2*3)))
        else:
            nail_solution_grids.append(get_centered_sq(grid_size, initial_sq_size-(2*(i-20))))
            
    return nail_solution_grids

