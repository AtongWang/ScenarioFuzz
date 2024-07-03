import sys

from carla import Transform,Location
from datetime import datetime





class FitnessExtractor:

    def __init__(self):
        self.ego_vehicle_location=None
        self.start_loc= None
        self.first= True
        self.i = 0
        self.first_dist_negative = True

        self.score_dict = {'dist_center_lane':[],'dist_min_other_vehicle':[],'dist_min_pedestrian':[],'dist_min_mesh':[],'dist_from_final_destnation':[]}

    def get_distance_from_center_lane(self, ego_vehicle, world):
        ego_vehicle_location = ego_vehicle.get_location()

        waypoint = world.get_map().get_waypoint(ego_vehicle_location, project_to_road=True)

        ego_vehicle_loc = Location(x=ego_vehicle_location.x, y=ego_vehicle_location.y, z=0.0)

        return ego_vehicle_loc.distance(waypoint.transform.location)

    def get_min_distance_from_other_vehicle(self,ego_vehicle, world):
 
        

        distances = [1000]
        ego_vehicle_location = ego_vehicle.get_location()

        for target_vehicle in world.get_actors().filter('vehicle.*'):
            distance = ego_vehicle_location.distance(target_vehicle.get_location())
            distances.append(distance)


        return (min(distances)) - 3.32 # substracting distances from center of vehicle

    def get_min_distance_from_pedestrians(self, ego_vehicle, world):
        distances = [1000]
        ego_vehicle_location = ego_vehicle.get_location()

        for target_vehicle in world.get_actors().filter('walker.*'):
            distance = ego_vehicle_location.distance(target_vehicle.get_location())
            distances.append(distance)

        return (min(distances)) -1.2  # substracting distances from center of vehicle

    def get_min_distance_from_static_mesh(self, ego_vehicle, world):
        distances = [1000]
        ego_vehicle_location = ego_vehicle.get_location()
        # for actor in world.get_actors():
        #     print(actor)
        for target_vehicle in world.get_actors().filter('static.*'):
            distance = ego_vehicle_location.distance(target_vehicle.get_location())
            distances.append(distance)

        for target_vehicle in world.get_actors().filter('traffic.*.*'):
            distance = ego_vehicle_location.distance(target_vehicle.get_location())
            distances.append(distance)
        for target_vehicle in world.get_actors().filter('traffic.*'):
            distance = ego_vehicle_location.distance(target_vehicle.get_location())
            distances.append(distance)

        return (min(distances))  # substracting distances from center of vehicle

    def get_distance_from_destination(self, ego_vehicle,dest_point_location):
        ego_vehicle_location = ego_vehicle.get_location()

        self.final_destination = dest_point_location
        # dist = ego_vehicle_location.distance(self.start_loc);
        dist = ego_vehicle_location.distance(self.final_destination)-10.8; #9.8 for v1

        return dist
    
    def extract_from_world(self,ego_vehicle,world,dest_point_location):


        dist_center_lane = self.get_distance_from_center_lane( ego_vehicle, world)
        dist_min_other_vehicle = self.get_min_distance_from_other_vehicle(ego_vehicle,world)
        dist_min_pedestrian= self.get_min_distance_from_pedestrians(ego_vehicle,world)
        dist_min_mesh=self.get_min_distance_from_static_mesh(ego_vehicle,world)
        dist_from_final_destnation=self.get_distance_from_destination(ego_vehicle,dest_point_location)

        self.score_dict['dist_center_lane'].append(dist_center_lane)
        self.score_dict['dist_min_other_vehicle'].append(dist_min_other_vehicle)
        self.score_dict['dist_min_pedestrian'].append(dist_min_pedestrian)
        self.score_dict['dist_min_mesh'].append(dist_min_mesh)
        self.score_dict['dist_from_final_destnation'].append(dist_from_final_destnation)

        if dist_from_final_destnation <0 : #to avoid logging while system is being shutdown
             if self.first_dist_negative == True:
                self.first_dist_negative = False 



    def get_values(self,state):
        DfC_min = 1
        DfV_min = 1
        DfP_min = 1
        DfM_min = 1
        DT_max = -1
        traffic_lights_max = 1

        # 检查是否有交通灯违规、车道侵犯和车辆碰撞的记录
        if state.red_violation:
            traffic_lights_max = 0
        if state.laneinvaded:
            DfC_min = 0
        if state.crashed:
            DfV_min = 0

        # 计算各项性能指标的最优值
        for i in range(len(self.score_dict['dist_center_lane'])):
            DfC = self.score_dict['dist_center_lane'][i]
            DfV = self.score_dict['dist_min_other_vehicle'][i]
            DfP = self.score_dict['dist_min_pedestrian'][i]
            DfM = self.score_dict['dist_min_mesh'][i]
            DT = self.score_dict['dist_from_final_destnation'][i]

            if DT < 0:
                DT_max = 1
                break

            DfC = 1 - (DfC / 1.15) # normalising
            DfV = min(DfV, 1)
            DfP = min(DfP, 1)
            DfM = min(DfM, 1)

            if i == 0:
                distance_Max = DT

            distance_travelled = distance_Max - DT
            normalised_distance_travelled = distance_travelled / distance_Max
            DfC_min = min(DfC_min, DfC)
            DfV_min = min(DfV_min, DfV)
            DfP_min = min(DfP_min, DfP)
            DfM_min = min(DfM_min, DfM)
            DT_max = max(DT_max, normalised_distance_travelled)

        return DfC_min, DfV_min, DfP_min, DfM_min, DT_max, traffic_lights_max

