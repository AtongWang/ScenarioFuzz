import os, sys, glob,json,random
import constants as c


def get_proj_root():
    config_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(config_path)
    proj_root = os.path.dirname(src_dir)

    return proj_root


def set_carla_api_path():
    proj_root = get_proj_root()

    dist_path = os.path.join(proj_root, "carla/PythonAPI/carla/dist")
    glob_path = os.path.join(dist_path, "carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64"
    ))

    try:
        api_path = glob.glob(glob_path)[0]
        
    except IndexError:
        print(glob_path)
        print("Couldn't set Carla API path.")
        exit(-1)

    if api_path not in sys.path:
        sys.path.append(api_path)
        print(f"API: {api_path}")


class Config:
    """
    A class defining fuzzing configuration and helper methods.
    An instance of this class should be created by the main module (fuzzer.py)
    and then be shared across other modules as a context handler.
    """

    def __init__(self):
        self.seed_counter = {town: {} for town in c.TOWN_LIST}
        self.debug = False
        self.strategy =None
        # simulator config
        self.sim_host = "localhost"
        self.sim_port = 2000
        self.sim_tm_port = 8000
        
        self.cov_name = None
        # Fuzzer config
        self.max_campaign_cnt = 0
        self.max_cycles = 0
        self.max_mutation = 0
        self.num_dry_runs = 1
        self.num_param_mutations = 1
        # self.initial_quota = 10

        # Fuzzing metadata
        self.cur_time = None
        self.time_list = None

        self.determ_seed = None
        self.out_dir = None
        self.seed_dir = None
        self.cache_dir = None
        self.scenario_data = None
        # Target config
        self.agent_type = c.AUTOWARE # c.AUTOWARE
        self.agent_sub_type = None
        # Enable/disable Various Checks
        self.check_dict = {
            "speed": True,
            "lane": False,
            "crash": True,
            "stuck": True,
            "red": False,
            "other": True,
        }
        self.test_type =None
        # Functional testing
        self.function = "general"

        # Sim-debug settings
        self.view = c.ONROOF

        self.reload_error_type = None
        self.reload_dir = None
        self.reload_target = None
        self.reload_target_error_dir = None

        self.cov_mode = False
        self.weather_by_name = False

        self.save_bag = False
    def set_paths(self):
        self.queue_dir = os.path.join(self.out_dir, "queue")
        self.error_dir = os.path.join(self.out_dir, "errors")
        self.cov_dir = os.path.join(self.out_dir, "cov")
        self.meta_file = os.path.join(self.out_dir, "meta")
        self.cam_dir = os.path.join(self.out_dir, "camera")
        self.rosbag_dir = os.path.join(self.out_dir, "rosbags")
        self.score_dir = os.path.join(self.out_dir, "scores")
        self.scenario_data = os.path.join(self.out_dir, "scenario_data")

    def increment_count(self,town,scenario_id):
        if scenario_id not in self.seed_counter[town]:
            self.seed_counter[town][scenario_id] = 1
        else:
            self.seed_counter[town][scenario_id] += 1
            
    def check_seed_entropy(self,campaign_cnt,seed):
        town_scenario_prob = {town: {scenario_id: count / campaign_cnt for scenario_id, count in scenario_dict.items()} for town, scenario_dict in self.seed_counter.items()}
        if seed['town'] in town_scenario_prob and seed['scenario_id'] in town_scenario_prob[seed['town']]:
            prob = town_scenario_prob[seed['town']][seed['scenario_id']]
            if random.random() <= prob:
                return False
            else:
                return True
        else:
            return True

    def enqueue_seed_scenarios(self):
        try:
            seed_scenarios = os.listdir(self.seed_dir)
        except:
            print("[-] Error - cannot find seed directory ({})".format(self.seed_dir))
            sys.exit(-1)

        queue = []
        for seed in seed_scenarios: 
            if not seed.startswith(".") and seed.endswith(".json"):
                seedfile = os.path.join(self.seed_dir, seed)
                with open(seedfile, "r") as fp:
                    seed = json.load(fp)    
                seed_info ={'town':seed['town'],'scenario_id':seed['scenario_id'],'scenario_type':seed['scenario_type']}      
                if self.town.lower() == 'all':           
                    queue.append(seed_info)
                elif self.town == seed['town']:
                    queue.append(seed_info)
                else:
                    pass
        return queue

    def reload_seed_scenarios(self):
        try:
            self.reload_dir_ads = os.path.join(self.reload_dir,self.reload_target.replace(':','-'),self.reload_error_type)
            seed_scenarios = os.listdir(self.reload_dir_ads)
        except:
            print("[-] Error - cannot find seed directory ({})".format(self.reload_dir_ads))
            sys.exit(-1)

        queue = []
        for seed_name in seed_scenarios: 
            if not seed_name.startswith(".") and seed_name.endswith(".json"):
                seedfile = os.path.join(self.reload_dir_ads, seed_name)
                with open(seedfile, "r") as fp:
                    seed = json.load(fp)
                    seed['file_name'] = seed_name      
                queue.append(seed)
        self.num_scenarios = len(queue)
        return queue