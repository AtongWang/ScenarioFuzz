import random

class Maneuver_unit:
    def __init__(self, direction, speed_or_degree, frame=0):
        self.direction = direction
        self.speed_or_degree = speed_or_degree
        self.frame = frame
        self.action_state = 0 # 0: not executed, 1: executed, 2: timeout
    
class Chromosome:
   

    def __init__(self, speed_bounds, action_bounds, NPC_size, time_size):
        self.y = 0
        self.scenario = [[[] for i in range(time_size)] for j in range(NPC_size)] # This scenario
        self.speed_bounds = speed_bounds
        self.action_bounds = action_bounds
        self.degree_bounds = [30,60]
        self.code_x1_length = NPC_size 
        self.code_x2_length = time_size
        self.timeoutTime = 300 # in seconds, timeout timer for simulator execution per each scenario simulation\
        self.NPC_size = NPC_size
        self.time_size = time_size
        
    def rand_init(self):
        self.scenario = []
        for n in range(self.NPC_size):
            npc_maneuver = []
            for t in range(self.time_size):
               # if t == 0:
                    #maneuver = Maneuver_unit(0,0,t) 
                # else:
                direction = random.randint(self.action_bounds[0], self.action_bounds[1])
                if direction == 0:
                    speed = random.uniform(self.speed_bounds[0], self.speed_bounds[1])
                    maneuver = Maneuver_unit(direction, speed, frame=t)
                else:
                    degree = random.uniform(self.degree_bounds[0], self.degree_bounds[1])
                    maneuver = Maneuver_unit(direction, degree, frame=t)
                npc_maneuver.append([maneuver.direction, maneuver.speed_or_degree, maneuver.action_state])
            self.scenario.append(npc_maneuver)
        return self.scenario
    
    def func(self,gen=None,lisFlag=False):
        pass

if __name__ == '__main__':
    NPC_size = 2
    time_size = 10
    speed_bounds = [0, 10]
    action_bounds = [-1, 1]
    chromosome = Chromosome(speed_bounds, action_bounds, NPC_size, time_size)
    chromosome.rand_init()
    print(chromosome.scenario)


            