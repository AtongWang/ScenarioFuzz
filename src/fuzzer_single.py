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
    argparser.add_argument("-t", "--target", default="leaderboard:Transfuser_V2", type=str,
            help="Target autonomous driving system (basic/behavior/autoware/leaderboard:(Transfuser/NEAT/LAV/))")
    argparser.add_argument("-f", "--function", default="general", type=str,
            choices=["general", "collision", "traction", "eval-os", "eval-us",
                "figure", "sens1", "sens2", "lat", "rear"],
            help="Functionality to test (general / collision / traction)")
    argparser.add_argument("--strategy", default="all", type=str,
            help="Input mutation strategy (all / congestion / entropy / instability / trajectory)")
    argparser.add_argument("--town", default='Town03', type=str,
            help="Test on a specific town (e.g., '--town 3' forces Town03),all will be random choose")
    
    argparser.add_argument("--scenario-id", default=0, type=int,
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
    return argparser


def scenario_fuzz(value_config,successor_scenario_value=None):
    sampler_class = sampler.ValueSampler(value_config)
    if successor_scenario_value is not None and not args.single_stage: 
        test_scenario_value = successor_scenario_value
        samples = sampler_class.sample(method=sampler.SamplingMethod.RANDOM_NEIGHBORS, num_samples=NUM_SAMPLE,reference_data=test_scenario_value) 
    else:    
        samples = sampler_class.sample(method=sampler.SamplingMethod.RANDOM, num_samples=NUM_SAMPLE) 

    return samples


def main():
    global args, conf, valuerangemanager, alarm_time, scene_id,campaign_cnt,scenario_dict,sc,NUM_SAMPLE
    conf = config.Config()
    argparser = set_args()
    args = argparser.parse_args()

    init(conf, args)

    alarm_time = args.alarm_time
    scene_id = 0
    campaign_cnt = 0

    signal.signal(signal.SIGALRM, handler)
    valuerangemanager = ValueRangeManager()
    sc = scenario_selector.ScenarioSelector()

    time_limit = args.time_limit
    test_start_time = time.time()

    if time_limit == 0:
        raise Exception("time_limit must be greater than 0")

    generate_method = args.method
    # if generate_method == "scenariofuzz":
    #     method_function = scenario_fuzz
    # else:
    #     raise Exception("method not found")
    
    while True:
        
        
        conf.time_list = {
        'campaign_start_time': None,
        'cycle_start_time': None,
        'mutation_start_time': None
        }

        campaign_start_time = datetime.datetime.now()
        conf.time_list['campaign_start_time'] = campaign_start_time.strftime('%Y-%m-%d %H:%M:%S')


        town_map = "{}".format(conf.town)
        sc.load_scenario_dict(args.scenario_lib_dir,town_map)

        scenario_id = args.scenario_id
        print("scenario_id:",scenario_id)
        scenario_dict = sc.get_scenario(str(scenario_id))
        #print(scenario_dict)
        scenario_type = scenario_dict['type']
        scenario_info = {'town':town_map,'scenario_id':scenario_id,'scenario_type':scenario_type}

        print("\n\033[1m\033[92m" + "=" * 10 + f"USING SCENARIO SEED:{scenario_info['town']} - {scenario_info['scenario_type']} - {scenario_info['scenario_id']}" + "=" * 10 + "\033[0m\n")
        
        test_scenario = fuzz_utils.TestScenario(conf)
        successor_scenario_value = None
        successor_scenario = None
        # STEP 3: SCENE MUTATION
        # While test scenario has quota left, mutate scene by randomly
        # generating walker/vehicle/puddle.

        cycle_cnt = 0
        while cycle_cnt < conf.max_cycles:
            cycle_start_time = datetime.datetime.now()
            conf.time_list['cycle_start_time'] = cycle_start_time.strftime('%Y-%m-%d %H:%M:%S')
            signal.alarm(conf.max_cycles * alarm_time * 60)  
            cycle_cnt += 1

            value_config = valuerangemanager.get_config()
            
            if generate_method == "scenariofuzz":
                samples = scenario_fuzz(value_config,successor_scenario_value)


            mutated_scenario_list = list() # mutated TestScenario objects
            score_list = list() # driving scores of each mutated scenario

            round_cnt = 0
    #-----------------------------------------------------------eval_model_predicted_samples-----------------------------------------------------------
            sample_index=0
            
            scenario_data_sample_list = []
            samples_data = []

            max_ov_num = len(sc.scenario_sp)-1

            device = args.device
            while sample_index < NUM_SAMPLE:
                sample = samples[sample_index]
                sc.G_init()

                s_,w_,d_ = sc.get_random_ego_car_start_end_point() 

                sc.add_scenario_to_graph()

                e_s_node = sc.get_nearest_node(s_)
                e_w_node = sc.get_nearest_node(w_)
                sc.G.nodes[e_s_node]['type'] = SC.EGO_CAR_SP
                sc.G.nodes[e_w_node]['type'] = SC.EGO_CAR_WP
                sc.G.edges[e_s_node,e_w_node]['way'] = SC.EGO_EDGE


                ov_list_input = min(len(sample['Ov']),max_ov_num)
                ov_list = sample['Ov'] =sample['Ov'][:ov_list_input]

                ov_wp_sp_list = sc.get_random_other_car_start_end_point_with_timeout(ov_list_input)

                
                for ov_attri,ov_s_w in zip(ov_list,ov_wp_sp_list):

                    v_type = ov_attri['Ov_type']
                    v_s = ov_s_w[0]
                    v_w = ov_s_w[1]
                    

                    if v_type == C.VEHICLE:
                        nav_type = ov_attri['Ov_behavior_v']
                        v_s_node = sc.get_nearest_node(v_s)
                        v_w_node = sc.get_nearest_node(v_w)
                        sc.G.nodes[v_s_node]['type'] = SC.OTHER_VEHICLE_SP
                        sc.G.nodes[v_s_node]['actor_name'] = ov_attri['Ov_name_v']
                        if nav_type != C.IMMOBILE:
                            sc.G.nodes[v_w_node]['type'] = SC.OTHER_VEHICLE_WP
                        else:
                            sc.G.edges[v_s_node,v_w_node]['way'] = SC.NO_PLAN

                    elif v_type == C.WALKER:
                        nav_type = ov_attri['Ov_behavior_w']
                        v_s_node = sc.get_nearest_node(v_s)
                        v_w_node = sc.get_nearest_node(v_w)
                        sc.G.nodes[v_s_node]['type'] = SC.OTHER_WALKER_SP
                        sc.G.nodes[v_s_node]['actor_name'] = ov_attri['Ov_name_w']
                        if nav_type != C.IMMOBILE:
                            sc.G.nodes[v_w_node]['type'] = SC.OTHER_WALKER_WP
                            sc.G.edges[v_s_node,v_w_node]['way'] = SC.OW_EDGE
                        else:
                            sc.G.edges[v_s_node,v_w_node]['way'] = SC.NO_PLAN
                weather_params = [sample[j] for j in weather_list]
                graph_data = sc.G2tensor(weather_param=weather_params)
                samples_data.append(graph_data)
                scenario_data_sample_list.append([sample,(s_,w_,d_),ov_wp_sp_list,sc.G])
                sample_index += 1
            eval_model = get_model(args.eval_model,device)
            test_loader = EdgeBatchDataLoader(samples_data,batch_size=NUM_SAMPLE, shuffle=False)

            with torch.no_grad():
                for batch in test_loader:
                    batch.to(device)
                    out = eval_model(batch.x, batch.edge_attr, batch.weather_attr, batch.edge_index, batch.batch, batch.edge_batch)
                    out_list = out.squeeze(-1).cpu().tolist()

            sorted_indices = sorted(range(len(out_list)), key=lambda x: out_list[x], reverse=True) #直接按照模型输出对所有样本进行排序
            choose_sample_list = [scenario_data_sample_list[index] for index in sorted_indices]
            choose_sample_index = 0
    #-----------------------------------------------------------eval_model_predicted_samples_end-----------------------------------------------------------

            while round_cnt < conf.max_mutations: # mutation rounds
                mutation_start_time = datetime.datetime.now()
                conf.time_list['mutation_start_time'] = mutation_start_time.strftime('%Y-%m-%d %H:%M:%S')
                signal.alarm(alarm_time * 60)
                round_cnt += 1
                
                print("\n\033[1m\033[92mCampaign #{} Cycle #{}/{} Mutation #{}/{}".format(
                    campaign_cnt, cycle_cnt, conf.max_cycles, round_cnt,
                    conf.max_mutations), "\033[0m", datetime.datetime.now())

                test_scenario_m = deepcopy(test_scenario)
                sample_data = choose_sample_list[choose_sample_index]

                s_,w_,d_ = sample_data[1]
                print('ego car set start_point:{}  end_point:{}   direction:{}'.format(s_,w_,SC.TO_PLAN_DRICTION[d_]))
                test_scenario_m.set_ego_wp_sp(s_,w_)


                sample = sample_data[0]
                ov_list = sample['Ov']
                ov_wp_sp_list = sample_data[2]
                            
                sc.G = sample_data[3]

                # STEP 3-1: ACTOR PROFILE GENERATION

                for ov_attri,ov_s_w in zip(ov_list,ov_wp_sp_list):
                    v_type = ov_attri['Ov_type']
                    v_s = ov_s_w[0]
                    v_w = ov_s_w[1]
                    v_d = ov_s_w[2]
                    dp_time = ov_attri['dp_time']
                    if v_type == C.VEHICLE:
                        v_name = C.VEHICLE_NAME_DICT[ov_attri['Ov_name_v']]
                        v_color = (ov_attri['Ov_R'],ov_attri['Ov_G'],ov_attri['Ov_B'])
                        nav_type = ov_attri['Ov_behavior_v']
                        vehicle_speed = ov_attri['Ov_speed_v']
                        ret = test_scenario_m.add_actor(v_type,
                                nav_type,v_s,v_w,vehicle_speed,v_name,v_color,dp_time)

                    elif v_type == C.WALKER:
                        w_name = C.WALKER_NAME_DICT[ov_attri['Ov_name_w']]
                        nav_type = ov_attri['Ov_behavior_w']
                        walker_speed = ov_attri['Ov_speed_w']
                        ret = test_scenario_m.add_actor(v_type,nav_type,v_s,v_w,walker_speed,w_name,dp_time=dp_time)

                    if conf.debug:
                        if v_type != C.NULL: 
                            print("[debug] successfully added {} {}".format(C.NAVTYPE_NAMES[nav_type], C.ACTOR_NAMES[v_type]))


                # STEP 3-2: PUDDLE PROFILE GENERATION
                puddle_list = sample['Puddle']
                for puddle_inf in puddle_list:
                    center_loc_x,center_loc_y,center_loc_z = scenario_dict['center_loc'][0],scenario_dict['center_loc'][1],scenario_dict['center_loc'][2]

                    level = puddle_inf['level']

                    x = puddle_inf['x_loc_size']
                    y = puddle_inf['y_loc_size']
                    z = puddle_inf['z_loc_size']

                    x_size =  puddle_inf['x_size']
                    y_size = puddle_inf['y_size']
                    z_size = puddle_inf['z_size']
                    loc = (center_loc_x+x, center_loc_y+y, center_loc_z+z) # carla.Location(x, y, z)
                    size = (x_size, y_size, z_size) # carla.Location(xy_size, xy_size, z_size)
                    ret = test_scenario_m.add_puddle(level, loc, size)

                    if conf.debug:
                        print("successfully added a puddle")

                # print("after seed gen and mutation", time.time())
                # STEP 3-3: EXECUTE SIMULATION
                ret = None
                #time.sleep(20)
                state = states.State()
                state.campaign_cnt = campaign_cnt
                state.cycle_cnt = cycle_cnt
                state.mutation = round_cnt
                state.file_name ="{}_{}_{}".format(state.campaign_cnt,state.cycle_cnt, state.mutation)
                
                state.device = args.device
                
                set_weather(test_scenario_m,sample) # mutate_weather_fixed(test)
                mutated_scenario_list.append(test_scenario_m) 
                
                exec_start_time = datetime.datetime.now()
                state.exec_start_time = exec_start_time.strftime('%Y-%m-%d %H:%M:%S')

                try:
                    ret = test_scenario_m.run_test(state)

                except Exception as e:
                    if e.args[0] == "HANG":
                        print("[-] simulation hanging. abort.")
                    else:
                        print("[-] run_test error:")
                    traceback.print_exc()
                    ret = -1
                signal.alarm(0)

                string = state.file_name+'-{}.json'.format( time.time())
                sc.save_json(os.path.join(conf.scenario_data, string),test_scenario_m.driving_quality_score,test_scenario_m.event_dict,sample)

                # t3 = time.time()
                choose_sample_index+=1
                # STEP 3-4: DECIDE WHAT TO DO BASED ON THE RESULT OF EXECUTION
                # TODO: map return codes with failures, e.g., spawn
                if ret is None:
                    # failure
                    pass

                elif ret == -1:
                    print("Spawn / simulation failure - don't add round cnt")
                    round_cnt -= 1

                elif ret == 1:
                    print("fuzzer - found an error")
                    # found an error - move on to next one in queue
                    # test.quota = 0
                    break

                elif ret == 128:
                    print("Exit by user request")
                    exit(0)

                elif ret == 666:
                    print("each the end point but systems no ending")

                else:
                #     if ret == -1:
                    print("[-] Fatal error occurred during test")
                    sys.exit(-1)

                score_list.append(test_scenario_m.driving_quality_score)
                ### mutation loop ends


            
            if test_scenario_m.found_error:
                # error detected. start a new cycle with a new seed
                successor_scenario_value = sample
                successor_scenario = test_scenario_m


            idx = score_list.index(min(score_list))
            successor_scenario_value = samples[idx]
            successor_scenario = mutated_scenario_list[idx]

            print(score_list)
            print(mutated_scenario_list)
            print("successor:", vars(successor_scenario))
            
            # shutil.copyfile(
            #     os.path.join(conf.queue_dir, successor_scenario.log_filename),
            #     os.path.join(conf.cov_dir, successor_scenario.log_filename)
            # )

            print("="*10 + " END OF ALL CYCLES " + "="*10)
            signal.alarm(0)

        
        campaign_cnt += 1
        

if __name__ == "__main__":
    main()
