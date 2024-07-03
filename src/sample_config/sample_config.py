import json,os,sys
sys.path.append('')
import src.constants as C
import scenario_select.scontant as SC

weather_list= ['cloud', 'rain', 'puddle', 'wind', 'fog', 'wetness', 'angle', 'altitude']
puddle_list = ['Puddle_num', 'level', 'x_loc_size', 'y_loc_size', 'z_loc_size', 'x_size', 'y_size', 'z_size']
ov_list = ['Ov_num','Ov_type','Ov_direction','Ov_behavior_v','Ov_behavior_w','Ov_speed_w','Ov_speed_v','Ov_name_w','Ov_name_v','Ov_R','Ov_G','Ov_B','dp_time']
value_ranges_set = {
    'Ego_direction':{'type': 'discrete', 'values':[k for k in SC.TO_PLAN_DRICTION.keys()]},
    'Ov_num':{'type': 'discrete', 'values': list(range(3,C.OV_MAX))},
    'Ov_type':{'type': 'discrete', 'values':C.ACTOR_LIST},
    'Ov_direction':{'type': 'discrete', 'values':[k for k in SC.TO_PLAN_DRICTION.keys()]},
    'Ov_behavior_v':{'type': 'discrete', 'values':C.NAVTYPE_LIST},
    'Ov_behavior_w':{'type': 'discrete', 'values':C.W_NAVTYPE_LIST},
    'Ov_name_v':{'type': 'discrete', 'values':[k for k in C.VEHICLE_NAME_DICT.keys()]},
    'Ov_name_w':{'type': 'discrete', 'values':[k for k in C.WALKER_NAME_DICT.keys()]},
    'Ov_R': {'type': 'continuous', 'min': 0, 'max': 255, 'step': 1},
    'Ov_G': {'type': 'continuous', 'min': 0, 'max': 255, 'step': 1},
    'Ov_B': {'type': 'continuous', 'min': 0, 'max': 255, 'step': 1},
    'dp_time':{'type': 'continuous', 'min': 15, 'max': 30, 'step': 1},
    'Ov_speed_w': {'type': 'continuous', 'min': 3, 'max': C.WALKER_MAX_SPEED, 'step': 1},
    'Ov_speed_v': {'type': 'continuous', 'min': 1, 'max': C.VEHICLE_MAX_SPEED, 'step': 1},
    'cloud': {'type': 'continuous', 'min': 0, 'max': 100, 'step': 1},
    'rain': {'type': 'continuous', 'min': 0, 'max': 100, 'step': 1},
    'puddle': {'type': 'continuous', 'min': 0, 'max': 100, 'step': 1},
    'wind': {'type': 'continuous', 'min': 0, 'max': 100, 'step': 1},
    'fog': {'type': 'continuous', 'min': 0, 'max': 100, 'step': 1},
    'wetness': {'type': 'continuous', 'min': 0, 'max': 100, 'step': 1},
    'angle': {'type': 'continuous', 'min': 0, 'max': 360, 'step': 1},
    'altitude': {'type': 'continuous', 'min': -90, 'max': 90, 'step': 1},
    'Puddle_num':{'type': 'discrete', 'values':list(range(C.PROB_PUDDLE))},
    'level':{'type': 'continuous',  'min': 0, 'max': 2, 'step': 0.1},
    'x_loc_size':{'type': 'continuous',  'min': 0, 'max': C.PUDDLE_X_Y_LOC_SIZE, 'step': 1},
    'y_loc_size':{'type': 'continuous',  'min': 0, 'max': C.PUDDLE_X_Y_LOC_SIZE, 'step': 1},
    'z_loc_size':{'type': 'continuous',  'min': 0, 'max': 0, 'step': 0.1},
    'x_size':{'type': 'continuous',  'min': 1, 'max': C.PUDDLE_X_Y_SIZE, 'step': 0.5},
    'y_size':{'type': 'continuous',  'min': 1, 'max': C.PUDDLE_X_Y_SIZE, 'step': 0.5},
    'z_size':{'type': 'continuous',  'min': 0, 'max': 0, 'step': 0.1}
}
class ValueRangeManager:
    def __init__(self, value_ranges=None):
        if not value_ranges:
            self.value_ranges = value_ranges_set
        else:
            self.value_ranges = value_ranges
        self.weather_list = weather_list
        self.puddle_list = puddle_list
        self.ov_list= ov_list
    def to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.value_ranges, f)

    def update(self, name, new_value_range):
        if name not in self.value_ranges:
            raise ValueError(f'Value range {name} not found')
        self.value_ranges[name].update(new_value_range)

    def add_discrete(self, name, choice):
        discrete_default = {'type': 'discrete','values':None}        
        if name in self.value_ranges:
            raise ValueError(f'Value range {name} already exists')
        discrete_default['values'] = choice 
        self.value_ranges[name] = discrete_default

    def add_discrete(self, name, min,max,step):
        continuous_default = {'type': 'continuous',  'min': 0, 'max': 0, 'step': 0}
        
        if name in self.value_ranges:
            raise ValueError(f'Value range {name} already exists')
        continuous_default['min'] = min 
        continuous_default['max'] = max
        continuous_default['step'] = step 
        self.value_ranges[name] = continuous_default
    
    def delete(self, name):
        if name not in self.value_ranges:
            raise ValueError(f'Value range {name} not found')
        del self.value_ranges[name]
        
    def get_config(self):
        return self.value_ranges


if __name__ =='__main__':

    manager = ValueRangeManager()