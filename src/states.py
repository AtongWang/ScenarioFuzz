""" Global States """
import signal
class State:
    """
    A state object stores raw data and events captured from the simulator,
    including vehicular states.
    Pass a reference to this object to TestScenario.run_test() as a parameter.
    """
    def __init__(self):
        # exec states
        self.loop = 0

        self.mutation = 0
        self.campaign_cnt = 0
        self.cycle_cnt = 0
        self.first_frame_id = 0
        self.first_sim_elapsed_time = 0
        self.sim_start_time = 0
        self.num_frames = 0
        self.elapsed_time = 0
        self._last_frame= 0

        self.file_name = None
        # failure states
        self.spawn_failed = False
        self.spawn_failed_object = None

        # error states
        self.crashed = False
        self.collision_event = None
        self.stuck = False
        self.stuck_duration = 0
        self.laneinvaded = False
        self.laneinvasion_event = []
        self.speeding = False
        self.speed = []
        self.speed_lim = []
        self.on_red = False
        self.on_red_speed = []
        self.red_violation = False

        # other error states, e.g., segfault
        self.other_error = False
        self.other_error_val = 0
        self.signal = 0

        # control states
        self.cont_throttle = []
        self.cont_brake = []
        self.cont_steer = []
        self.plan_wp=[]
        self.steer_angle_list = []
        self.yaw_list = []
        self.yaw_rate_list = []
        self.lat_speed_list = []
        self.lon_speed_list = []
        self.min_dist_r_list = []
        self.min_dist = 99999
        self.object_trajectory = {}
        # score
        self.deductions = {}

        # debug info
        self.autoware_cmd = ""
        self.autoware_goal = ""

        # time log
        self.exec_start_time = 0
        self.exec_end_time = 0
        self.exec_scenario_finish_time = 0

        self.weather ={}

        self.fitness_cal_object =  None
        
    def sig_handler(self, signum, frame):
        print("[-] something happened: {}".format(signal.signum.name))
        self.other = True
        self.signal = signum
    
    def get_time(self):
        """
        Returns elapsed game time
        """
        return self.elapsed_time


    def get_frame(self):
        """
        Returns elapsed game time
        """
        return self._last_frame
