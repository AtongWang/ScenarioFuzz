"""Define constants used across ScenriaoFuzz modules."""
TOWN_LIST =['Town01','Town02','Town03','Town04','Town05']
MSG_BAD_WEATHER_ARG = "Weather argument should be in range [0.0:100.0]."
MSG_BAD_SUN_ARG = "Sun args require --angle [0:360], --altitude [-90:90]"
MSG_EMPTY_FRICTION_ARG = "--friction should be followed by level, size " \
                         "and location coordinate."
MSG_BAD_FRICTION_ARG = "Number of friction args has to be 7."
MSG_BAD_FRICTION_LEVEL = "Friction level should be in range [0.0-1.0]."
MSG_EMPTY_ACTOR_ARG = "--actor should be followed by actor_type, and " \
                      "corresponding values."
MSG_BAD_ACTOR_TYPE = "--actor expects 0, 1, or 2 as actor_type."
MSG_BAD_ACTOR_ATTR = "Vehicle or walker requires 12 args: " \
                     "actor_type, nav_type, sp_x, sp_y, sp_z, " \
                     "sp_pitch, sp_yaw, sp_roll, dp_x, dp_y, dp_z, " \
                     "speed."
MSG_BAD_SPAWN_ARG = "--spawn expects six args: x, y, z, pitch, yaw, roll"
MSG_BAD_DEST_ARG = "--dest expects three args: x, y, z"


#WALKER_NAME_DICT = { 1: 'walker.pedestrian.0001'}
WALKER_NAME_DICT = { 1: 'walker.pedestrian.0001', 2: 'walker.pedestrian.0002', 3: 'walker.pedestrian.0003', 4: 'walker.pedestrian.0004', 5: 'walker.pedestrian.0005', 6: 'walker.pedestrian.0006', \
                    7: 'walker.pedestrian.0007', 8: 'walker.pedestrian.0008', 9: 'walker.pedestrian.0009', 10: 'walker.pedestrian.0010', 11: 'walker.pedestrian.0011', \
                    12: 'walker.pedestrian.0012', 13: 'walker.pedestrian.0013', 14: 'walker.pedestrian.0014'}
# 15: 'walker.pedestrian.0015', 16: 'walker.pedestrian.0016',\
# 17: 'walker.pedestrian.0017', 18: 'walker.pedestrian.0018', 19: 'walker.pedestrian.0019', 20: 'walker.pedestrian.0020', 21: 'walker.pedestrian.0021', \
# 22: 'walker.pedestrian.0022', 23: 'walker.pedestrian.0023', 24: 'walker.pedestrian.0024', 25: 'walker.pedestrian.0025', 26: 'walker.pedestrian.0026'
# VEHICLE_NAME_DICT={0: 'vehicle.audi.a2', 1: 'vehicle.audi.tt', 2: 'vehicle.carlamotors.carlacola', 3: 'vehicle.bmw.isetta', 4: 'vehicle.nissan.micra', \
#                 5: 'vehicle.citroen.c3', 6: 'vehicle.gazelle.omafiets', 7: 'vehicle.mercedes-benz.coupe', 8: 'vehicle.mini.cooperst',\
#                 9: 'vehicle.nissan.patrol', 10: 'vehicle.mustang.mustang', 11: 'vehicle.lincoln.mkz2017', 12: 'vehicle.tesla.cybertruck',\
#                 13: 'vehicle.toyota.prius', 14: 'vehicle.volkswagen.t2', 15: 'vehicle.bmw.grandtourer', 16: 'vehicle.tesla.model3',\
#                 17: 'vehicle.diamondback.century', 18: 'vehicle.dodge_charger.police', 19: 'vehicle.bh.crossbike', 20: 'vehicle.kawasaki.ninja', \
#                 21: 'vehicle.jeep.wrangler_rubicon', 22: 'vehicle.yamaha.yzf', 23: 'vehicle.chevrolet.impala', 24: 'vehicle.harley-davidson.low_rider', \
#                 25: 'vehicle.audi.etron', 26: 'vehicle.seat.leon'}
VEHICLE_NAME_DICT={
0: "vehicle.tesla.model3",
1: "vehicle.audi.a2",
2: "vehicle.bh.crossbike",
3: "vehicle.bmw.grandtourer",
4: "vehicle.carlamotors.carlacola",
5: "vehicle.chevrolet.impala",
6: "vehicle.dodge_charger.police",
7: "vehicle.jeep.wrangler_rubicon",
8: "vehicle.lincoln.mkz2017",
9: "vehicle.nissan.micra",
10: "vehicle.seat.leon",
11: "vehicle.tesla.cybertruck",
12: "vehicle.toyota.prius",
13: "vehicle.volkswagen.t2"
}
# VEHICLE_NAME_DICT={
# 0: "vehicle.tesla.model3"}
# Agent Types
BASIC    = 0
BEHAVIOR = 1
AUTOWARE = 2
LEADERBOARD=3
L_SYSTEM_DICT = {'transfuser':0,'neat':1,'lav':2,'lav_v2':3,'transfuser_v2':4}
OTHER    = 9

# BasicAgent config
TARGET_SPEED = 60
MAX_THROTTLE = 1 # 0.75
MAX_BRAKE = 1 #0.3
MAX_STEERING = 0.8

# Static configurations
MAX_DIST_FROM_PLAYER = 40
MIN_DIST_FROM_PLAYER = 5
FRAME_RATE = 20
INIT_SKIP_SECONDS = 2
WAIT_AUTOWARE_NUM_TOPICS = 195

# Actors
NULL    = -1 # special type for traction testing
VEHICLE = 0
WALKER  = 1
ACTOR_LIST = [VEHICLE, WALKER]
ACTOR_NAMES = ["vehicle", "walker"]

# Actor Navigation Type
LINEAR    = 0
AUTOPILOT = 1
IMMOBILE  = 2
MANEUVER  = 3
W_NAVTYPE_LIST = [LINEAR, AUTOPILOT, IMMOBILE]
NAVTYPE_LIST = [LINEAR, AUTOPILOT, IMMOBILE,MANEUVER]
NAVTYPE_NAMES = ["linear", "autopilot", "immobile", "maneuver"]

# Actor Attributes
VEHICLE_MAX_SPEED = 30 # multiplied with forward vector
WALKER_MAX_SPEED = 10 # m/s

# Puddle Attributes
PROB_PUDDLE = 25 # probability of adding a new puddle
PUDDLE_MAX_SIZE = 500 # centimeters
PUDDLE_X_Y_LOC_SIZE =25
PUDDLE_X_Y_SIZE =5

# Maneuver Attributes
FRAMES_PER_TIMESTEP = 100 # 5 seconds (tentative)

# Number of waypoints per town
NUM_WAYPOINTS = {
        "Town01": 255,
        "Town02": 101,
        "Town03": 265,
        "Town04": 372,
        "Town05": 302,
        "Town06": 436,
    }

# Camera View Setting
ONROOF   = 0
BIRDSEYE = 1

# Driving Quality
HARD_ACC_THRES = 21.2 # km/h per second
HARD_BRAKE_THRES = -21.2 # km/h per second

# Filter config
CUTOFF_FREQ_LIGHT = 3.5
CUTOFF_FREQ_HEAVY = 0.5

# Mutation targets
WEATHER = 0
ACTOR   = 1
PUDDLE  = 2
MUTATION_TARGET = [WEATHER, ACTOR, PUDDLE]

# Input mutation strategies
ALL = 0
CONGESTION = 1
ENTROPY = 2
INSTABILITY = 3
TRAJECTORY = 4

# Misc
DEVNULL = "2> /dev/null"


OV_MAX = 6
MIN_DIST_TO_GOAL=2
MAX_RUN_TIME=20