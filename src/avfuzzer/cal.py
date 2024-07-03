import math



maxint =130


def brakeDist(speed):
    dBrake = 0.0467 * pow(speed, 2.0) + 0.4116 * speed - 1.9913 + 0.5
    if dBrake < 0:
        dBrake = 0
    return dBrake

def calculate_deltaD(ego_trajectory, vehicles_trajectories,ego_speed_list):
    deltaD_values = []

    for i in range(len(ego_trajectory)):
        ego = ego_trajectory[i]
        min_deltaD = maxint
        ego_speed = ego_speed_list[i]

        for vehicle_trajectory in vehicles_trajectories:
            if i < len(vehicle_trajectory):  # Check if the trajectory has a state at this time step
                npc = vehicle_trajectory[i]
                d = findDistance(ego, npc) - 4.6

                deltaDFront = maxint
                deltaDSide = maxint

                # Check if npc is in front of ego

                npc_x, npc_y, npc_z = npc
                ego_x, ego_y, ego_z = ego
                if npc_x > ego_x and npc_x < ego_x + 20:
                    if npc_y > ego_y - 2 and npc_y < ego_y + 2:
                        deltaDFront = d - brakeDist(ego_speed)
                # Check if ego is changing lane to npc's front
                if npc_x < ego_x and npc_x > ego_x - 20:
                    if npc_y > ego_y - 2 and npc_y < ego_y + 2:
                        deltaDFront = d - brakeDist(ego_speed)
                # Calculate the minimum deltaD for this npc
                min_deltaD_npc = min(deltaDSide, deltaDFront)
                # Update the global minimum deltaD if necessary
                min_deltaD = min(min_deltaD, min_deltaD_npc)

        deltaD_values.append(min_deltaD)

    return deltaD_values


def calculate_D(ego_trajectory, vehicles_trajectories):

    D_values = []

    for i in range(len(ego_trajectory)):
        ego = ego_trajectory[i]
        min_D = maxint

        for vehicle_trajectory in vehicles_trajectories:
            if i < len(vehicle_trajectory):  # Check if the trajectory has a state at this time step
                npc = vehicle_trajectory[i]
                d = findDistance(ego, npc)

                min_D = min(min_D, d)

        D_values.append(min_D)

    return D_values

def findDistance(ego,npc):
    ego_x, ego_y, ego_z = ego
    npc_x, npc_y, npc_z = npc
    dis= math.pow((ego_x-npc_x),2)+math.pow((ego_y-npc_y),2)+math.pow((ego_z-npc_z),2)
    dis= math.sqrt(dis)
    return dis

def findFitness(deltaD_values, D_values):
    # 计算 minDeltaD
    minDeltaD = min(deltaD_values) if deltaD_values else maxint

    # 计算 minD
    minD = min(D_values) if D_values else maxint


    # 计算适应度值
    fitness = 0.5 * minD + 0.5 * minDeltaD

    ## 如果是 ego 车的过失，适应度值可能需要调整

    fitness *= -1  # 或者根据您的需求进行其他形式的调整

    return fitness

def cal_fitness(ego_car_trajectory, other_vehicles_trajectories, ego_speed,npc_num):
    deltaD_over_time = calculate_deltaD(ego_car_trajectory, other_vehicles_trajectories,ego_speed)
    D_over_time = calculate_D(ego_car_trajectory, other_vehicles_trajectories)
    score = findFitness(deltaD_over_time, D_over_time)

    fitness = (score+maxint)/float(npc_num-1)
    return fitness
# ego_car_trajectory = state.object_trajectory['ego_car']['trajectory']
# ego_speed =state.speed
# other_vehicles_trajectories = [v['trajectory'] for v in state.object_trajectory['vehicles']]
# deltaD_over_time = calculate_deltaD(ego_car_trajectory, other_vehicles_trajectories)
