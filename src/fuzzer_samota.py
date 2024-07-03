#!/usr/bin/env python3

import os
import sys
import time
import random
import argparse
import json
from collections import deque
import docker
import subprocess as sp
import math
import datetime
import signal
from copy import deepcopy
import traceback
import shutil
import warnings

import config


import constants as C
import states
import executor
from sample_config.sample_config import ValueRangeManager
import sampler 
import fuzz_utils

sys.path.append('.')
import scenario_select.scenario_selector as scenario_selector
import scenario_select.scontant as SC

from scenario_eval_model.data_load import EdgeBatchDataLoader
from scenario_eval_model.eval_model_list import get_model


from src.samota.fitness_value_extractor import FitnessExtractor
from src.samota.samota import run_search



# try:
import carla
# except ModuleNotFoundError as e:
#     print("[-] Carla module not found. Make sure you have built Carla.")
#     proj_root = config.get_proj_root()
#     print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
#     exit(-1)
import torch
weather_list= ['cloud', 'rain', 'puddle', 'wind', 'fog', 'wetness', 'angle', 'altitude']

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning)

# 忽略所有警告
warnings.filterwarnings('ignore')

def handler(signum, frame):
    print("[-] 代码运行超时，结束运行。")
    raise Exception("HANG")

def timeout_handler(signum, frame):
    print("[-] 代码运行超时，结束运行。")
    sys.exit(-1)
    

def init(conf, args):
    global NUM_SAMPLE
    ''' 
    参数赋值于conf类,创建文件夹
    
    '''
    # Fuzzing 开始时间
    NUM_SAMPLE = args.mutation_num
    conf.cur_time = time.time()
    if args.determ_seed:
        conf.determ_seed = args.determ_seed
    else:
        conf.determ_seed = conf.cur_time
    random.seed(conf.determ_seed)
    print("[info] determ seed set to:", conf.determ_seed)

    target_dir = args.out_dir+'/{}'.format(args.target).replace(':','-')
    os.makedirs(target_dir,exist_ok=True)

    now = datetime.datetime.now()
    time_now = now.strftime("%Y-%m-%d-%H-%M")
    out_dir = os.path.join(target_dir,'out-artifact-{}'.format(time_now))
    #print(target_dir,out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    os.mkdir(out_dir)
    conf.out_dir = out_dir
    print("[info] all information wiil be stored at:",conf.out_dir)
    
    if args.seed_dir is None:
        conf.seed_dir = os.path.join(target_dir,'seed-artifact')
    else:
        conf.seed_dir = args.seed_dir

   
    # if  args.target.lower() != "autoware":
    conf.cache_dir = args.cache_dir    
    # else:
    #     conf.cache_dir = '/tmp/fuzzerdata'
        
    # if os.path.exists(conf.cache_dir ):
    #     shutil.rmtree(conf.cache_dir )
    if not os.path.exists(conf.cache_dir):
        os.mkdir(conf.cache_dir)
    print(f"Using cache dir {conf.cache_dir}")

    if not os.path.exists(conf.seed_dir):
        os.mkdir(conf.seed_dir)
    else:
        print(f"Using seed dir {conf.seed_dir}")

    if args.verbose:
        conf.debug = True
    else:
        conf.debug = False

    conf.set_paths()

    with open(conf.meta_file, "w") as f:
        f.write(" ".join(sys.argv) + "\n")
        f.write("start: " + str(int(conf.cur_time)) + "\n")

    try:
        os.mkdir(conf.queue_dir)
        os.mkdir(conf.error_dir)
        os.mkdir(conf.cov_dir)
        os.mkdir(conf.rosbag_dir)
        os.mkdir(conf.cam_dir)
        os.mkdir(conf.score_dir)
        os.mkdir(conf.scenario_data)
    except Exception as e:
        print(e)
        sys.exit(-1)

    conf.sim_host = args.sim_host
    conf.sim_port = args.sim_port
    conf.sim_tm_port = conf.sim_port + 50
    conf.max_cycles = args.max_cycles
    conf.max_mutations = args.max_mutations

    conf.timeout = args.timeout

    conf.function = args.function

    if args.target.lower() == "basic":
        conf.agent_type = C.BASIC
    elif args.target.lower() == "behavior":
        conf.agent_type = C.BEHAVIOR
    elif args.target.lower() == "autoware":
        conf.agent_type = C.AUTOWARE
    elif 'leaderboard' in args.target.lower():
        l_system = args.target.split(':')[1].lower()
        l_id = C.L_SYSTEM_DICT[l_system]
        conf.agent_type = C.LEADERBOARD
        conf.agent_sub_type = l_id
    else:
        print("[-] Unknown target: {}".format(args.target))
        sys.exit(-1)
    #若添加新系统，需从此入门修改。
    conf.town = args.town

    if args.no_speed_check:
        conf.check_dict["speed"] = False
    if args.no_crash_check:
        conf.check_dict["crash"] = False
    if args.no_lane_check:
        conf.check_dict["lane"] = False
    if args.no_stuck_check:
        conf.check_dict["stuck"] = False
    if args.no_red_check:
        conf.check_dict["red"] = False
    if args.no_other_check:
        conf.check_dict["other"] = False

    if args.strategy == "all":
        conf.strategy = C.ALL
    elif args.strategy == "congestion":
        conf.strategy = C.CONGESTION
    elif args.strategy == "entropy":
        conf.strategy = C.ENTROPY
    elif args.strategy == "instability":
        conf.strategy = C.INSTABILITY
    elif args.strategy == "trajectory":
        conf.strategy = C.TRAJECTORY
    else:
        print("[-] Please specify a strategy")
        exit(-1)

    if args.coverage:
        conf.cov_mode = True

    if args.weather_by_name:
        conf.weather_by_name = True

def mutate_weather(test_scenario):
    test_scenario.weather["cloud"] = random.randint(0, 100)
    test_scenario.weather["rain"] = random.randint(0, 100)
    test_scenario.weather["puddle"] = random.randint(0, 100)
    test_scenario.weather["wind"] = random.randint(0, 100)
    test_scenario.weather["fog"] = random.randint(0, 100)
    test_scenario.weather["wetness"] = random.randint(0, 100)
    test_scenario.weather["angle"] = random.randint(0, 360)
    test_scenario.weather["altitude"] = random.randint(-90, 90)

def set_weather(test_scenario,weather_param):
    test_scenario.weather["cloud"] = weather_param["cloud"]
    test_scenario.weather["rain"] =  weather_param["rain"]
    test_scenario.weather["puddle"] = weather_param["puddle"]
    test_scenario.weather["wind"] =  weather_param["wind"]
    test_scenario.weather["fog"] =  weather_param["fog"]
    test_scenario.weather["wetness"] =  weather_param["wetness"]
    test_scenario.weather["angle"] =  weather_param["angle"]
    test_scenario.weather["altitude"] =  weather_param["altitude"]


def mutate_weather_fixed(test_scenario):
    test_scenario.weather["cloud"] = 0
    test_scenario.weather["rain"] = 0
    test_scenario.weather["puddle"] = 0
    test_scenario.weather["wind"] = 0
    test_scenario.weather["fog"] = 0
    test_scenario.weather["wetness"] = 0
    test_scenario.weather["angle"] = 0
    test_scenario.weather["altitude"] = 60


def set_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--out-dir", default="/workspace2/scenario_fuzz_model", type=str,
            help="Directory to save fuzzing logs")
    argparser.add_argument("--cache-dir", default="/workspace1/fuzzerdata", type=str,
    help="store fuzz cache data")
    argparser.add_argument("--scenario-lib-dir", default="scenario_lib", type=str,
    help="store scenario lib  data")
    argparser.add_argument("-s", "--seed-dir", default=None, type=str,
            help="Seed directory")
    argparser.add_argument("-c", "--max-cycles", default=3, type=int,
            help="Maximum number of loops")
    argparser.add_argument("-m", "--max-mutations", default=3, type=int,
            help="Size of the mutated population per cycle")
    argparser.add_argument("--mutation-num",default=100,type=int,help="Size of the mutated population per cycle")
    argparser.add_argument("-d", "--determ-seed", type=float,
            help="Set seed num for deterministic mutation (e.g., for replaying)")
    argparser.add_argument("-v", "--verbose", action="store_true",
            default=False, help="enable debug mode")
    argparser.add_argument("--direction-set", action="store_true",
        default=False, help="enable debug mode")
    argparser.add_argument("-u", "--sim-host", default="localhost", type=str,
            help="Hostname of Carla simulation server")
    argparser.add_argument("-p", "--sim-port", default=2000, type=int,
            help="RPC port of Carla simulation server")
    argparser.add_argument("-t", "--target", default="leaderboard:LAV_V2", type=str,
            help="Target autonomous driving system (basic/behavior/autoware/leaderboard:(Transfuser/NEAT/LAV/))")
    argparser.add_argument("-f", "--function", default="general", type=str,
            choices=["general", "collision", "traction", "eval-os", "eval-us",
                "figure", "sens1", "sens2", "lat", "rear"],
            help="Functionality to test (general / collision / traction)")
    argparser.add_argument("--strategy", default="all", type=str,
            help="Input mutation strategy (all / congestion / entropy / instability / trajectory)")
    argparser.add_argument("--town", default='Town03', type=str,
            help="Test on a specific town (e.g., '--town 3' forces Town03),all will be random choose")
    
    argparser.add_argument("--scenario-id", default=1, type=int,
            help="Test on a specific scenario id (e.g., '--scenario-id 3' forces scenario 3)")
    
    argparser.add_argument("--timeout", default="30", type=int,
            help="Seconds to timeout if vehicle is not moving")
    argparser.add_argument("-a","--alarm-time", default="10", type=int,
        help="Seconds to timeout if vehicle is not moving")
    
    argparser.add_argument("--eval-model", default="improve-v2", type=str,
        help="set_eval_model_type",choices= ["improve","basic","improve-v1","improve-v2"])
    argparser.add_argument("--device", default="cuda:0", type=str,
        help="device to run the model(default:cuda:0/1,cpu)")
    argparser.add_argument("--no-use-seed",action="store_true",help="no use seed in first cycle")
    argparser.add_argument("--no-speed-check", action="store_true")
    argparser.add_argument("--no-lane-check", action="store_true")
    argparser.add_argument("--no-crash-check", action="store_true")
    argparser.add_argument("--no-stuck-check",default=False,action="store_true")
    argparser.add_argument("--no-red-check", action="store_true")
    argparser.add_argument("--no-other-check",default=True, action="store_true")
    argparser.add_argument("--single-stage",default=False,action="store_true")
    argparser.add_argument("--coverage",default=False,action="store_true")
    argparser.add_argument("--time-limit",default=6,type=int,help="run time limit,unit:hours")
    argparser.add_argument("--method", default="scenariofuzz", type=str,
        help="choose generation method",choices= ["scenariofuzz","avfuzz","autofuzz","samota","behavxplore"])
    argparser.add_argument("--weather-by-name",action="store_true",default=True)
    return argparser

class Pylot_caseStudy():
    def __init__(self,scenario_id,test_scenario):
        
        self.scenario_id = scenario_id
        self.lib = {1:{'straight':"samota_lib/Town03_T_1_straight.json",
                       'left':"samota_lib/Town03_T_1_left.json",
                        'right':"samota_lib/Town03_T_1_right.json"},
                    3:{'straight':"samota_lib/Town03_straight_3.json"},
                    6:{'straight':"samota_lib/Town03_cross_6_straight.json",
                        'left':"samota_lib/Town03_cross_6_left.json",
                        'right':"samota_lib/Town03_cross_6_right.json"}}
        self.scenario_type_lib = {1:'T-intersection',
                                    3:'Straight',
                                    6:'Cross'}
        self.scenario_dict = self.lib[scenario_id]

        self.scenario_type = self.scenario_type_lib[scenario_id]
        self.route_info = None

        self.test_scenario_m = None
        self.npcs_start_point = None

        self.test_scenario = test_scenario

        self.campaign_cnt = 0
        self.cycle_cnt = 0
        self.mutation = 0

    def add_actor(self,actor_position_type,is_2_wheeled):

        if actor_position_type == 'Vehicle_in_front':
            actor_index = 0
        elif actor_position_type == 'vehicle_in_adjcent_lane':
            actor_index = 1
        elif actor_position_type == 'vehicle_in_opposite_lane':
            actor_index = 2
        else:
            actor_index = 0

        actor_type = C.VEHICLE
       
        
        if is_2_wheeled:
            npc_name = 'vehicle.bh.crossbike'
            nav_type = C.LINEAR   # 2wheeled autopilot exist bug,so use linear
            speed = 5
        else:
            npc_name = 'vehicle.bmw.grandtourer'
            nav_type = C.AUTOPILOT
            speed = 0

        npc_start_point = self.npcs_start_point[actor_index]['start_point']

        npc_s = [(npc_start_point['x'],npc_start_point['y'],max(2,npc_start_point['z']+1)),
        (npc_start_point['roll'],npc_start_point['pitch'],npc_start_point['yaw'])]
                    
        ret = self.test_scenario_m.add_actor(actor_type,nav_type,npc_s,None,speed,name=npc_name,specific_waypoints=True,dp_time=10)


    def set_weather(self,fv):
        if (fv[8] == 0):  # noon
            if (fv[9] == 0):  # clear
                weather = "ClearNoon"
            if (fv[9] == 1):  # clear
                weather = "CloudyNoon"
            if (fv[9] == 2):  # clear
                weather = "WetNoon"
            if (fv[9] == 3):  # clear
                weather = "WetCloudyNoon"
            if (fv[9] == 4):  # clear
                weather = "MidRainyNoon"
            if (fv[9] == 5):  # clear
                weather = "HardRainNoon"
            if (fv[9] == 6):  # clear
                weather = "SoftRainNoon"
        if (fv[8] == 1):  # sunset
            if (fv[9] == 0):  # clear
                weather = "ClearSunset"
            if (fv[9] == 1):  # clear
                weather = "CloudySunset"
            if (fv[9] == 2):  # clear
                weather = "WetSunset"
            if (fv[9] == 3):  # clear
                weather = "WetCloudySunset"
            if (fv[9] == 4):  # clear
                weather = "MidRainSunset"
            if (fv[9] == 5):  # clear
                weather = "HardRainSunset"
            if (fv[9] == 6):  # clear
                weather = "SoftRainSunset"
        if (fv[8] == 2):  # sunset
            if (fv[9] == 0):  # clear
                weather = "ClearSunset"
            if (fv[9] == 1):  # clear
                weather = "CloudySunset"
            if (fv[9] == 2):  # clear
                weather = "WetSunset"
            if (fv[9] == 3):  # clear
                weather = "WetCloudySunset"
            if (fv[9] == 4):  # clear
                weather = "MidRainSunset"
            if (fv[9] == 5):  # clear
                weather = "HardRainSunset"
            if (fv[9] == 6):  # clear
                weather = "SoftRainSunset"

            weather = weather+'_night_time'
        
        self.test_scenario_m.weather = weather

    def add_pedestrian(self,fv):

        if fv[10] == 0:
            num_of_pedestrians = 0
        if fv[10] == 1:
            num_of_pedestrians = 5
        else:
            num_of_pedestrians = 0
        

        if num_of_pedestrians == 0:
            return
        else:
            opposite_wp = self.npcs_start_point[2]['start_point']
            opposite_wp_tuple = ((opposite_wp['x'],opposite_wp['y'],opposite_wp['z']),(opposite_wp['roll'],opposite_wp['pitch'],opposite_wp['yaw']))

            ego_car_opposite_sidebike = self.test_scenario_m.get_locations_sidebike(opposite_wp_tuple,num_of_pedestrians)

            if len(ego_car_opposite_sidebike)>0:
                for people_transform in ego_car_opposite_sidebike:
                    people_transform_tuple = [(people_transform['x'],people_transform['y'],people_transform['z']),(people_transform['roll'],people_transform['pitch'],people_transform['yaw'])]
                    self.test_scenario_m.add_actor(C.WALKER,C.IMMOBILE,people_transform_tuple,None,0,name='walker.pedestrian.0001',dp_time=10,specific_waypoints=True)




    def construct_scenario(self,fv):
        '''
        # 0 Direction road
        # 1 Scenario Length
        # 2 Vehicle_in_front
        # 3 vehicle_in_adjcent_lane
        # 4 vehicle_in_opposite_lane
        # 5 vehicle_in_front_two_wheeled
        # 6 vehicle_in_adjacent_two_wheeled
        # 7 vehicle_in_opposite_two_wheeled
        # 8 time of day 1
        # 9 weather  1
        # 10 Number of People 
        '''

        self.test_scenario_m = deepcopy(self.test_scenario)


        if fv[1] == 0 or self.scenario_type == 'Straight':
            scenario_file = self.scenario_dict['straight']
        elif fv[1] == 1 and not self.scenario_type == 'Straight':
            scenario_file = self.scenario_dict['left']
        elif fv[1] == 2 and not self.scenario_type == 'Straight':
            scenario_file = self.scenario_dict['right']
        
        with open(scenario_file) as f:
            self.route_info = json.load(f)
        
        ego_car_start_point = self.route_info['ego']['start_point']

        scenario_lenth_index = fv[2]
        middle_point_lenth = len(self.route_info['ego']['middle_point'])
        if scenario_lenth_index > middle_point_lenth-1:
            ego_car_end_point = self.route_info['ego']['end_point']
        else:
            ego_car_end_point = self.route_info['ego']['middle_point'][scenario_lenth_index]

        ego_s = [(ego_car_start_point['x'],ego_car_start_point['y'],max(2,ego_car_start_point['z']+1)),
        (ego_car_start_point['roll'],ego_car_start_point['pitch'],ego_car_start_point['yaw'])]
        ego_w = [(ego_car_end_point['x'],ego_car_end_point['y'],max(2,ego_car_end_point['z']+1)),
        (ego_car_end_point['roll'],ego_car_end_point['pitch'],ego_car_end_point['yaw'])]

        self.test_scenario_m.set_ego_wp_sp(ego_s,ego_w,specific_waypoints=True)

        self.npcs_start_point = self.route_info['other_vehicle']
        # Set Vehicle or 2-wheeled vehicle
        if fv[3] == 1:
            self.add_actor('Vehicle_in_front',fv[5])
        if fv[4] == 1:
            self.add_actor('vehicle_in_adjcent_lane',fv[6])
        if fv[5] == 1:
            self.add_actor('vehicle_in_opposite_lane',fv[7])
        
        # Weather
        self.set_weather(fv)
        
        self.add_pedestrian(fv)



    def _evaluate(self,x):

        fv = x
        self.construct_scenario(fv)

        signal.alarm(alarm_time * 60)
        # 执行场景
        ret = None
        #time.sleep(20)
        state = states.State()
        state.campaign_cnt = self.campaign_cnt
        state.cycle_cnt = self.cycle_cnt
        state.mutation = self.mutation
        state.file_name ="{}_{}_{}".format(state.campaign_cnt,state.cycle_cnt, state.mutation)
        
        state.device = args.device  

        exec_start_time = datetime.datetime.now()
        state.exec_start_time = exec_start_time.strftime('%Y-%m-%d %H:%M:%S')

        state.fitness_cal_object = FitnessExtractor()

        try:
            ret = self.test_scenario_m.run_test(state)

        except Exception as e:
            if e.args[0] == "HANG":
                print("[-] simulation hanging. abort.")
            else:
                print("[-] run_test error:")
            traceback.print_exc()
            ret = -1
        signal.alarm(0)


        # STEP 3-4: DECIDE WHAT TO DO BASED ON THE RESULT OF EXECUTION
        # TODO: map return codes with failures, e.g., spawn
        if ret is None:
            # failure
            #pass
            print("[-] Fatal error occurred during test")

        elif ret == -1:
            print("Spawn / simulation failure - don't add round cnt")

        elif ret == 1:
            print("fuzzer - found an error")
            # found an error - move on to next one in queue
            # test.quota = 0
            # break

        elif ret == 128:
            print("Exit by user request")
            exit(0)

        elif ret == 666:
            print("each the end point but systems no ending")

        else:
        #     if ret == -1:
            print("[-] Fatal error occurred during test")
            sys.exit(-1)

        DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max  = state.fitness_cal_object.get_values(state)

        score_data_info ={'fv':fv,
                        'DfC_min':DfC_min,
                        'DfV_max':DfV_max,
                        'DfP_max':DfP_max,
                        'DfM_max':DfM_max,
                        'DT_max':DT_max,
                        'traffic_lights_max':traffic_lights_max,
                        'ret':ret,
        }

        score_file_name = os.path.join(conf.scenario_data,state.file_name+'-{}.json'.format(time.time()))
        with open(score_file_name,'w') as f:
            json.dump(score_data_info,f)
        
        return [DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max]


def main():
    global args, conf, alarm_time
    conf = config.Config()
    argparser = set_args()
    args = argparser.parse_args()

    init(conf, args)

    alarm_time = args.alarm_time
    campaign_cnt = 0

    signal.signal(signal.SIGALRM, handler)

    town_map = "{}".format(conf.town)

    scenario_id = args.scenario_id


    print("\n\033[1m\033[92m" + "=" * 10 + f"USING SCENARIO SEED:{town_map} -  {scenario_id}" + "=" * 10 + "\033[0m\n")


    test_scenario = fuzz_utils.TestScenario(conf)


    #-------------SAMOTA-CONFIG-------------
    pop_size = 6  # 6
    lb = [0,0,0,0,0,0,0,0,0,0,0]
    ub = [3,3,2,2,2,2,2,2,3,7,2]
    #ub = [3,3,1,1,1,1,1,1,3,7,2]

    '''
    # 0 Direction road
    # 1 Scenario Length
    # 2 Vehicle_in_front
    # 3 vehicle_in_adjcent_lane
    # 4 vehicle_in_opposite_lane
    # 5 vehicle_in_front_two_wheeled
    # 6 vehicle_in_adjacent_two_wheeled
    # 7 vehicle_in_opposite_two_wheeled
    # 8 time of day 1
    # 9 weather  1
    # 10 Number of People 
    '''
    
    
    threshold_criteria = [0,0,0,0,0.95,0]
    no_of_Objectives = 6
    g_max = 200
    
    #-------------SAMOTA-CONFIG-END-------------


    while True:
        
        
        conf.time_list = {
        'campaign_start_time': None,
        'cycle_start_time': None,
        'mutation_start_time': None
        }

        campaign_start_time = datetime.datetime.now()
        conf.time_list['campaign_start_time'] = campaign_start_time.strftime('%Y-%m-%d %H:%M:%S')

        archive = []
        database = []
        
        #-------------SAMOTA-START-------------
        scenario_case = Pylot_caseStudy(scenario_id,test_scenario)

        run_search(scenario_case._evaluate,pop_size,lb,ub,no_of_Objectives,threshold_criteria,archive,database,g_max,conf.out_dir)
               
        campaign_cnt += 1
        

if __name__ == "__main__":
    main()
