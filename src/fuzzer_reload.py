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

#config.set_carla_api_path()
# try:
import carla
# except ModuleNotFoundError as e:
#     print("[-] Carla module not found. Make sure you have built Carla.")
#     proj_root = config.get_proj_root()
#     print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
#     exit(-1)

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
    

def set_determ_seed(seed,conf):
    conf.determ_seed = seed
    random.seed(seed)
    print("[info] determ seed set to:", seed)
    
def init(conf, args):
    ''' 
    参数赋值于conf类,创建文件夹
    
    '''
    conf.cur_time = time.time()
    conf.reload_error_type = args.error_type
    conf.reload_dir = args.reload_dir
    conf.reload_target = args.target

    target_dir = args.out_dir+'/{}'.format(args.target).replace(':','-')
    target_dir_error = os.path.join(target_dir,args.error_type) 
    if not os.path.exists(target_dir_error):
        os.makedirs(target_dir_error,exist_ok=True)
    conf.reload_target_error_dir = target_dir_error

def init_single(conf,args,scenario_info):
    is_exist = False
    set_determ_seed(scenario_info['determ_seed'],conf)

    file_name = scenario_info['file_name'].split('.json')[0]
    out_dir = os.path.join(conf.reload_target_error_dir,'out-artifact-{}'.format(file_name))
    #print(target_dir,out_dir)
    if os.path.exists(out_dir) and len(os.listdir(os.path.join(out_dir,'camera'))) > 0:
        is_exist = True
        print("[info] reload file {} already exist, skip it.".format(out_dir))
        return is_exist
    elif os.path.exists(out_dir) and len(os.listdir(os.path.join(out_dir,'camera'))) == 0:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir,exist_ok=True)
    else:
        os.makedirs(out_dir,exist_ok=True)
        
    conf.out_dir = out_dir
    print("[info] all information wiil be stored at:",conf.out_dir)
    

    conf.town = scenario_info['town']
    # if  args.target.lower() != "autoware":
    conf.cache_dir = args.cache_dir    
    # else:
    #     conf.cache_dir = '/tmp/fuzzerdata'
        
    if os.path.exists(conf.cache_dir):
        shutil.rmtree(conf.cache_dir)

    os.mkdir(conf.cache_dir)
    print(f"Using cache dir {conf.cache_dir}")



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


    conf.timeout = args.timeout


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
    return is_exist

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
    argparser.add_argument("-o", "--out-dir", default="/workspace2/scenario_reload", type=str,
            help="Directory to save fuzzing logs")
    argparser.add_argument("--cache-dir", default="/workspace1/fuzzerdata", type=str,
    help="store fuzz cache data")

    
    argparser.add_argument("--error-type", default="crash", type=str,help='reload error type scenario')
    argparser.add_argument("--reload-dir", default="/workspace2/scenario_data_reload", type=str,help='reload dir')

    argparser.add_argument("-v", "--verbose", action="store_true",
            default=False, help="enable debug mode")

    argparser.add_argument("-u", "--sim-host", default="localhost", type=str,
            help="Hostname of Carla simulation server")
    argparser.add_argument("-p", "--sim-port", default=20000, type=int,
            help="RPC port of Carla simulation server")
    argparser.add_argument("-t", "--target", default="leaderboard:LAV", type=str,
            help="Target autonomous driving system (basic/behavior/autoware/leaderboard:(Transfuser/NEAT/LAV/))")
    
    
    argparser.add_argument("--town", default='Town03', type=str,
            help="Test on a specific town (e.g., '--town 3' forces Town03),all will be random choose")
    
    argparser.add_argument("--timeout", default="30", type=int,
            help="Seconds to timeout if vehicle is not moving")

    argparser.add_argument("--device", default="cuda", type=str,
        help="eval_model_run_device")
    argparser.add_argument("--no-speed-check", action="store_true")
    argparser.add_argument("--no-lane-check", action="store_true")
    argparser.add_argument("--no-crash-check", action="store_true")
    argparser.add_argument("--no-stuck-check",default=False,action="store_true")
    argparser.add_argument("--no-red-check", action="store_true")
    argparser.add_argument("--no-other-check",default=True, action="store_true")

    return argparser


def main():
    conf = config.Config()
    argparser = set_args()
    args = argparser.parse_args()

    init(conf, args)

    seed_queue = deque(conf.reload_seed_scenarios())

    scene_id = 0

    ret = None
    signal.signal(signal.SIGALRM, handler)
    max_iter = 3
    iter_count = 0
    while scene_id < conf.num_scenarios:
        signal.alarm(int(args.timeout)* 60)
        
        if seed_queue.__len__() == 0:
            break
        scenario_information = seed_queue.popleft()
        is_exist = init_single(conf,args,scenario_information)

        if is_exist and not ret == -1:
            scene_id += 1
            continue
        print("\n\033[1m\033[92m" + "=" * 10 + f"SCENARIO ID: {scene_id}/{conf.num_scenarios} RELOAD: {args.target} ERROR TYPE:{conf.reload_error_type} \
              FILE NAME:{scenario_information['file_name']} " + "=" * 10 + "\033[0m\n")

            



        # STEP 2: TEST CASE INITIALIZATION
        # Create and initialize a TestScenario instance based on the metadata
        # read from the popped seed file.

        
        
        successor_scenario_value = None
        successor_scenario = None
        
        # STEP 3: TEST CASE EXECUTION
        # STEP 3.1: SET EGO CAR PATH

        ego_sp = scenario_information['ego_sp']
        ego_dp = scenario_information['ego_dp']
        if ego_sp is None or ego_dp is None:
            scene_id += 1
            continue

        test_scenario = fuzz_utils.TestScenario(conf)
        test_scenario.log_filename = scenario_information['file_name']
        test_scenario.set_ego_wp_sp(ego_sp,ego_dp)
        print('ego car set start point:{} destination point:{}'.format(ego_sp,ego_dp))
        # STEP 3.2 :RELOAD ADD ACTORS
        
        actors_information = scenario_information['actors']

        for actor in actors_information:
            actor_sp = actor['sp']
            actor_dp = actor['dp']
            actor_type = actor['type']
            actor_nav_type = actor['nav_type']
            actor_speed = actor['speed']
            actor_name = actor['name']
            actor_color = actor['color'] if 'color' in actor else None
            actor_dp_time = actor['dp_time']
            ret = test_scenario.add_actor(actor_type,actor_nav_type,actor_sp,actor_dp,actor_speed,actor_name,actor_color,actor_dp_time)

        # STEP 3.3: PUDDLE PROFILE GENERATION
        puddle_list = scenario_information['puddles']
        for puddle_inf in puddle_list:

            level = puddle_inf['level']

            x = puddle_inf['sp_x']
            y = puddle_inf['sp_y']
            z = puddle_inf['sp_z']

            x_size =  puddle_inf['size_x']
            y_size = puddle_inf['size_y']
            z_size = puddle_inf['size_z']
            loc = (x, y, z) # carla.Location(x, y, z)
            size = (x_size, y_size, z_size) # carla.Location(xy_size, xy_size, z_size)
            ret = test_scenario.add_puddle(level, loc, size)

            if conf.debug:
                print("successfully added a puddle")

        # STEP 3.4: SET WEATHER
        weather = scenario_information['weather']
        set_weather(test_scenario, weather)

        # STEP 3.5: EXECUTE SIMULATION
        

        #time.sleep(20)
        state = states.State()
 
        state.file_name  =  scenario_information['file_name'].split('.json')[0]

                
        try:
            ret = test_scenario.run_test(state)

        except Exception as e:
            if e.args[0] == "HANG":
                print("[-] simulation hanging. abort.")
            else:
                print("[-] run_test error:")
            traceback.print_exc()
            ret = -1
        
        signal.alarm(0)

        

        scene_id += 1

        # STEP 3-4: DECIDE WHAT TO DO BASED ON THE RESULT OF EXECUTION
        # TODO: map return codes with failures, e.g., spawn
        if ret is None:
            # failure
            pass

        elif ret == -1:
            print("Spawn / simulation failure - don't add round cnt")
            if iter_count < max_iter:
                seed_queue.append(scenario_information)
                scene_id -= 1
                iter_count += 1
                print(f"reload and error reappear, iter_count:{iter_count}/{max_iter}")
            else:
                iter_count = 0
                print(f"not add round cnt, next will reload")
                continue
        elif ret == 1:
            print("reload and error reappear")


        elif ret == 128:
            print("Exit by user request")
            exit(0)

        elif ret == 666:
            print("each the end point but systems no ending")

        else:
        #     if ret == -1:
            print("[-] Fatal error occurred during test")
            sys.exit(-1)

        

    print("Task completed")


if __name__ == "__main__":
    main()
