import random
from typing import List, Dict, Any
import random
from enum import Enum
import itertools
import numpy as np
import math
from decimal import Decimal

weather_list= ['cloud', 'rain', 'puddle', 'wind', 'fog', 'wetness', 'angle', 'altitude']
puddle_list = ['level', 'x_loc_size', 'y_loc_size', 'z_loc_size', 'x_size', 'y_size', 'z_size']
ov_list = ['Ov_type','Ov_direction','Ov_behavior_v','Ov_behavior_w','Ov_speed_w','Ov_speed_v','Ov_name_w','Ov_name_v','Ov_R','Ov_G','Ov_B','dp_time']

def random_choice(min_val, max_val, step):
    n = int((max_val - min_val) / step) + 1
    index = random.randrange(n)
    value = Decimal(str(min_val)) + index * Decimal(str(step))
    return float(value)


class SamplingMethod(Enum):
    RANDOM = 'random'
    GRID = 'grid'
    RANDOM_NEIGHBORS = 'random_neighbors'

class ValueType(Enum):
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
    DEFAULT = 'default'

class ValueSampler:
    def __init__(self, value_ranges):
        if not value_ranges:
            raise ValueError('Value ranges cannot be empty')
        self.value_ranges= {key: value_ranges[key] for key in value_ranges.keys() if key not in puddle_list and key not in ov_list}
        self.puddle_ranges = {key: value_ranges[key] for key in value_ranges.keys() if key in puddle_list}
        self.ov_ranges =  {key: value_ranges[key] for key in value_ranges.keys() if key in ov_list}

    def sample(self, method: SamplingMethod = SamplingMethod.RANDOM, **kwargs):
        if method == SamplingMethod.RANDOM:
            return self._sample_random(**kwargs)
        elif method == SamplingMethod.GRID:
            return self._sample_grid(**kwargs)
        elif method == SamplingMethod.RANDOM_NEIGHBORS:
            return self._sample_random_neighbors(**kwargs)
        else:
            raise ValueError(f'Unknown sampling method {method}')

    def _sample_random(self,num_samples: int = 1,value_ranges = None,choosed=False):
        """
        采样方法：随机采样
        """
        if value_ranges:
            value_ranges = value_ranges
        else:
            value_ranges = self.value_ranges
        values_list = []
        for s in range(num_samples):
            values = {}
            for name, value_range in value_ranges.items():
                if value_range['type'] == ValueType.DISCRETE.value:
                    values[name] = random.choice(value_range['values'])
                    if not choosed:
                        if name == 'Ov_num' :
                            if values[name]!=0:
                                values['Ov'] = self._sample_random(values[name],self.ov_ranges,True)
                            else:
                                values['Ov'] =[]
                        if name == 'Puddle_num' :
                            if values[name]!=0:
                                values['Puddle'] = self._sample_random(values[name],self.puddle_ranges,True)
                            else:
                                values['Puddle']=[]
                elif value_range['type'] == ValueType.CONTINUOUS.value: 
                    values[name] = random_choice(value_range['min'], value_range['max'],value_range['step']) 
            values_list.append(values)
        return values_list

    def _sample_grid(self, num_samples: int = 1,value_ranges = None): 
        """ 采样方法：网格采样 """
        if value_ranges:
            value_ranges = value_ranges
        else:
            value_ranges = self.value_ranges

        def calculate_x(sample_len, sample_num):
            """
            计算 x^sample_len >= sample_num 的最小整数 x
            """
            x = math.ceil(sample_num ** (1/sample_len))
            return x
        def split_list(lst, n):
            length = len(lst)
            sub_length = length // n
            remainder = length % n
            result = []
            start = 0
            for i in range(n):
                if i < remainder:
                    end = start + sub_length + 1
                else:
                    end = start + sub_length
                result.append(lst[start])
                start = end
            return result

        values_list = [] 
        value_ranges_list = list(value_ranges.items()) 
        num_ranges = len(value_ranges_list) 
        num_grids = calculate_x(num_ranges,num_samples)


        # 生成网格点
        grid_points = []
        for i in range(num_ranges):
            name, value_range = value_ranges_list[i]
            if value_range['type'] == ValueType.DISCRETE.value:
                grid_points.append(split_list(value_range['values'],num_grids))
            elif value_range['type'] == ValueType.CONTINUOUS.value:
                grid_points.append(np.linspace(value_range['min'], value_range['max'], num_grids))


        # 生成网格采样点
        for idx in itertools.product(*[range(len(gp)) for gp in grid_points]):
            values = {}
            for i, (name, value_range) in enumerate(value_ranges_list):
                if value_range['type'] == ValueType.DISCRETE.value:
                    values[name] = grid_points[i][idx[i]]
                    if name == 'Ov_num' and values[name]!=0:
                        values['Ov'] = self._sample_grid(values[name],self.ov_ranges)
                    else:
                        values['Ov']=[]
                    if name == 'Puddle_num'and values[name]!=0:
                        values['Puddle'] = self._sample_grid(values[name],self.puddle_ranges)
                    else:
                        values['Puddle']=[]
                elif value_range['type'] == ValueType.CONTINUOUS.value:
                    values[name] = grid_points[i][idx[i]]
            values_list.append(values) 
        values_list = values_list[:num_samples]
        return values_list



    def _sample_random_neighbors(self, num_samples: int = 1, value_range_step: int = 5, reference_data:Dict[str, Any] = None, value_ranges = None):
        """
        采样方法：随机邻居采样
        """
        if reference_data is not None:
            center_values = reference_data
        else:
            return None

        if value_ranges:
            value_ranges = value_ranges
        else:
            value_ranges = self.value_ranges
            center_values = {key: center_values[key] for key in center_values.keys() if key not in ['Ov','Puddle']+[k for k in self.ov_ranges.keys()]+[k for k in self.puddle_ranges.keys()]}
            puddle_center_values =reference_data['Puddle']
            ov_center_values = reference_data['Ov']
        values_list = []

        for s in range(num_samples):
            values = {}
            for name, value_range in value_ranges.items():
                if value_range['type'] == ValueType.DISCRETE.value:
                    v= center_values[name]
                    if name not in ['Ov_num','Puddle_num']:
                        v_index = value_range['values'].index(v)
                        values[name] = random.choice(value_range['values'][max(0,v_index-value_range_step):min(len(value_range['values']),v_index+value_range_step)])
                    elif name == 'Ov_num':
                        values[name] = center_values[name]
                        values['Ov'] = []
                        if values[name]!=0:
                            for index in range(values['Ov_num']):
                                values['Ov'].append(self._sample_random_neighbors(num_samples=1,value_ranges=self.ov_ranges,reference_data=ov_center_values[index])[0])
                    elif name == 'Puddle_num':
                        values[name] = center_values[name]
                        values['Puddle'] = []
                        if values[name]!=0:
                            for index in range(values['Puddle_num']):
                                values['Puddle'].append(self._sample_random_neighbors(num_samples=1,value_ranges=self.puddle_ranges,reference_data=puddle_center_values[index])[0])
                elif value_range['type'] == ValueType.CONTINUOUS.value:
                    v = center_values[name]
                    values[name] = random_choice(max(v - value_range_step*value_range['step'], value_range['min']),
                                                min(v + value_range_step*value_range['step'], value_range['max']),value_range['step'])
            values_list.append(values)
        return values_list

    def __len__(self):
        """
        计算采样空间的大小
        """
        total = 1
        for value_range in self.value_ranges.values():
            if value_range['type'] == ValueType.DISCRETE.value:
                total *= len(value_range['values'])
            elif value_range['type'] == ValueType.CONTINUOUS.value:
                total *= int((value_range['max'] - value_range['min']) / value_range['step']) + 1
        return total