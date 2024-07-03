import random
import copy
import os
import pickle
from datetime import datetime
import collections

from src.avfuzzer.chromosome import Chromosome, Maneuver_unit


class GeneticAlgorithm:
    def __init__(self, speed_bounds, action_bounds, pop_size, NPC_size, time_size, max_gen, pm, pc,Chromosome_object,campaign):
        self.speed_bounds = speed_bounds
        self.action_bounds = action_bounds
        self.degree_bounds = [30,60]
        self.pop_size = pop_size
        self.NPC_size = NPC_size
        self.time_size = time_size
        self.max_gen = max_gen
        self.pop = []
        self.bests = [0] * max_gen
        self.bestIndex = 0
        self.g_best = None
        self.ck_path = None
        self.touched_chs = []
        self.isInLis = False
        self.minLisGen = 2
        self.numOfGenInLis = 5
        self.hasRestarted = False
        self.lastRestartGen = 0
        self.bestYAfterRestart = 0

        self.pm = pm
        self.pc = pc

        self.chromosome = Chromosome_object
        self.campaign = campaign

    def setLisFlag(self):
        self.isInLis = True

    def init_pop(self):
        for i in range(self.pop_size):
            chromosome = self.chromosome(self.speed_bounds, self.action_bounds, self.NPC_size, self.time_size)
            chromosome.campaign_cnt = self.campaign
            chromosome.mutation = i
            chromosome.rand_init()
            chromosome.func(0)
            self.pop.append(chromosome)

    def cross(self):
        # Implementation of random crossover

        for i in range(int(self.pop_size / 2.0)):
            # Check crossover probability
            if self.pc > random.random():
            # randomly select 2 chromosomes(scenarios) in pops
                i = 0
                j = 0
                while i == j:
                    i = random.randint(0, self.pop_size-1)
                    j = random.randint(0, self.pop_size-1)
                pop_i = self.pop[i]
                pop_j = self.pop[j]

                # Record which chromosomes have been touched
                self.touched_chs.append(self.pop[i])
                self.touched_chs.append(self.pop[j])

                # Every time we only switch one NPC between scenarios
                # select cross index
                swap_index = random.randint(0, pop_i.code_x1_length - 1)

                temp = copy.deepcopy(pop_j.scenario[swap_index])
                pop_j.scenario[swap_index] = copy.deepcopy(pop_i.scenario[swap_index])
                pop_i.scenario[swap_index] = temp


    def mutation(self,gen):
        i = 0
        while(i<len(self.pop)) :
            eachChs = self.pop[i]
            i += 1
            if self.pm >= random.random():
                

                beforeMutation = copy.deepcopy(eachChs)
                # select mutation index
                npc_index = random.randint(0, eachChs.code_x1_length-1)
                time_index = random.randint(0, eachChs.code_x2_length-1)

                # Record which chromosomes have been touched
                self.touched_chs.append(eachChs)
                actionIndex = random.randint(0, 1)
                
                if actionIndex == 0:
                    # Change action_type 
                    action_type = random.randint(self.action_bounds[0], self.action_bounds[1])
                    
                    eachChs.scenario[npc_index][time_index][0] = action_type
                    if action_type == 0:
                        speed = random.uniform(self.speed_bounds[0], self.speed_bounds[1])
                        eachChs.scenario[npc_index][time_index][1] = speed
                    else:
                        degree = random.uniform(self.degree_bounds[0], self.degree_bounds[1])
                        eachChs.scenario[npc_index][time_index][1] = degree

                elif actionIndex == 1:
                    # Change speed or degree
                    if eachChs.scenario[npc_index][time_index][0] == 0:
                        speed = random.uniform(self.speed_bounds[0], self.speed_bounds[1])
                        eachChs.scenario[npc_index][time_index][1] = speed
                    else:
                        degree = random.uniform(self.degree_bounds[0], self.degree_bounds[1])
                        eachChs.scenario[npc_index][time_index][1] = degree


            # Only run simulation for the chromosomes that are touched in this generation
            if eachChs in self.touched_chs:
                eachChs.func(gen, self.isInLis)
            # else:
            #     util.print_debug(" --- The chromosome has not been touched in this generation, skip simulation. ---")


            # util.print_debug(" --- In mutation: Current scenario has y = " + str(eachChs.y))
        
        # def ga(gen):

        #     if not self.isInLis:
        #         self.init_pop()
            
        #     self.g_best = copy.deepcopy(best)

        #     print(" --- In generation: " + str(gen) + " ---")

        #     self.touched_chs = []
        #     self.cross()
        #     self.mutation()
        #     best, bestIndex = self.find_best()
        #     self.bests[i] = best


    def select_roulette(self):

        sum_f = 0


        for i in range(0, self.pop_size):
            if self.pop[i].y == 0:
                self.pop[i].y = 0.001

        min = self.pop[0].y
        for k in range(0, self.pop_size):
            if self.pop[k].y < min:
                min = self.pop[k].y
        if min < 0:
            for l in range(0, self.pop_size):
                self.pop[l].y = self.pop[l].y + (-1) * min

        # roulette
        for i in range(0, self.pop_size):
            sum_f += self.pop[i].y
        p = [0] * self.pop_size
        for i in range(0, self.pop_size):
            if sum_f == 0:
                sum_f = 1
            p[i] = self.pop[i].y / sum_f
        q = [0] * self.pop_size
        q[0] = 0
        for i in range(0, self.pop_size):
            s = 0
            for j in range(0, i+1):
                s += p[j]
            q[i] = s

        # start roulette
        v = []
        for i in range(0, self.pop_size):
            r = random.random()
            if r < q[0]:
                selectedChromosome = self.chromosome(self.speed_bounds,self.action_bounds, self.NPC_size, self.time_size)
                selectedChromosome.scenario = self.pop[0].scenario
                selectedChromosome.y = self.pop[0].y
                v.append(selectedChromosome)
            for j in range(1, self.pop_size):
                if q[j - 1] < r <= q[j]:
                    selectedChromosome = self.chromosome(self.speed_bounds,self.action_bounds,self.NPC_size, self.time_size)
                    selectedChromosome.scenario = self.pop[j].scenario
                    selectedChromosome.y = self.pop[j].y
                    v.append(selectedChromosome)
        self.pop = copy.deepcopy(v)

    def setLisPop(self, singleChs):
        for i in range(self.pop_size):
            self.pop.append(copy.deepcopy(singleChs))

        # Add some entropy
        tempPm = self.pm
        self.pm = 1
        self.mutation(0)
        self.pm = tempPm
        self.g_best, bestIndex = self.find_best()




    def find_best(self):
        best = copy.deepcopy(self.pop[0])
        bestIndex = 0
        for i in range(self.pop_size):
            if best.y < self.pop[i].y:
                best = copy.deepcopy(self.pop[i])
                bestIndex = i
        return best, bestIndex
    
    def take_checkpoint(self, obj, ck_name,target_dir=None):
        if target_dir is None:
            dir = 'GaCheckpoints/'
        else:
            dir = os.path.join(target_dir, 'GaCheckpoints/')
        if os.path.exists(dir) == False:
            os.mkdir(dir)
        ck_f = open(dir + ck_name, 'wb')
        pickle.dump(obj, ck_f)
        ck_f.truncate() 
        ck_f.close()