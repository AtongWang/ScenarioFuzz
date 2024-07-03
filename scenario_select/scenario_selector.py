import json,os,sys
import random,math
import networkx as nx
import numpy as np
import copy
sys.path.append('scenario_select')
import scontant as S
sys.path.append('src')
import torch
from torch_geometric.data import Data
from timeout_decorator import timeout, TimeoutError
def to_local_coordinate(wp, target_wp): 
    x, y, z = target_wp 
    new_x = wp[0] - x 
    new_y = wp[1] - y 
    new_z = wp[2] - z 
    return (new_x, new_y, new_z)
    
def most_frequent_element(lst):
    if len(lst) ==1:
        return lst[0]
    elif len(lst) >1:
        lst = [[item,lst.count(item)] for item in lst]  # 将列表转换为元组
        return max(lst, key=lambda x: x[1])[0]
    else:
        return None
class ScenarioSelector:

    def __init__(self):
        self.file_dir=None
        self.town = None
        self.scearnio_id = None
        self.ego_car_wp = None
        self.ego_car_sp = None
        self.ego_car_directin =None
        self.other_w_s_d_list ={}
        self.G = nx.Graph()
        self.scearnio_type = None
    @staticmethod
    def change_scearnio_file(file_path,town):
        new_scenario_dict = {}
        with open(os.path.join(file_path,f'{town}_scenario_lib.json')) as f:
            scenario_dict = json.load(f)
        id = 0
        for index,scenario in enumerate(scenario_dict.values()):
            new_scenario = ScenarioSelector.get_scenario_reverse(scenario,change_file=True)
            if len(new_scenario['wp']) != 0 and len(new_scenario['plan_way'])>10:
                new_scenario_dict[id] = new_scenario
                id+=1
        new_file_path = os.path.join(os.path.join(file_path,f'{town}_scenario_lib_new.json'))
                # 保存为json文件
        # 假设 scenario_dict_tf 是一个字典，其中包含 set 类型的值
        with open(new_file_path, 'w') as f_json:
            json.dump(new_scenario_dict, f_json)
            

    @staticmethod
    def get_scenario_reverse(scenario,change_file=False):

        scenario_way = [way for way in scenario['plan_way'] if way[0] != 'Unknown' and way[1]>5]
        scenario_way_list = []
        for way in scenario_way:
            route_wp = way[2]
            for wp in route_wp:
                if wp not in scenario_way_list and wp in scenario['wp']:
                    scenario_way_list.append(wp)
        if not change_file:
            if len(scenario_way_list) == 0:
                return False
            else:
                return True
        else:
            index = []
            for wp in scenario_way_list:
                index.append(scenario['wp'].index(wp))
            scenario['wp'] = scenario_way_list
            scenario['wp_mark'] = [scenario['wp_mark'][i] for i in index]
            scenario['wp_lanetype'] = [scenario['wp_lanetype'][i] for i in index]
            scenario['plan_way'] = scenario_way
            return scenario

    def load_scenario_dict(self,file_dir,town):
        self.town = town
        self.file_dir =file_dir
        try:
            with open(os.path.join(file_dir,f'{town}_scenario_lib_new.json')) as f:
                self.scenario_dict = json.load(f)
        except:
            print(f'No scenario file found for town {town}.')
        self.scearnio_type_list = list(set([v['type'] for v in self.scenario_dict.values()]))

    def wp2index(self, wp):
        if not isinstance(wp,list):
            return self.scenario['wp'].index(list(wp))  
        else:
            return self.scenario['wp'].index(wp)


    def get_scenario_type(self, scenario_type, traffic_light=None, traffic_sign=None):
        # Filter scenarios by type and traffic light/sign

        self.scearnio_type = scenario_type
        filtered_scenarios = [(key, value) for key, value in self.scenario_dict.items()
                              if value['type'] == scenario_type and
                              (traffic_light is None or len(value['tf_loc']) != 0) and
                              (traffic_sign is None or traffic_sign in value['wp_mark'])]
        
        # Randomly select a scenario and waypoint
        if len(filtered_scenarios) == 0:
            return None
        else:
            return  filtered_scenarios


    def get_scenario(self, scenario_key):
        self.scearnio_id = scenario_key
        self.scenario = self.scenario_dict[self.scearnio_id]
        self.scenario_node = []
        self.scenario_sp = []
        for n in self.scenario['plan_way']:
            dir_, length, path, reverse = n
            for w in path:
                if w not in [x[0] for x in self.scenario_node]:
                    if w in self.scenario['wp']:
                        self.scenario_node.append((w, self.wp2index(w)))
                    else:
                        self.scenario_node.append((w, None))
            for sp in path[:-1]:
                min_dist_list =[self.get_edge_weight(sp, x) for x in self.scenario_sp]
                min_edge_weight = min(min_dist_list) if len(min_dist_list)!=0 else 6
                if sp not in self.scenario_sp and min_edge_weight > 3:
                    self.scenario_sp.append(sp)
        self.node_list = [node[0] for node in self.scenario_node]
        return self.scenario  

    def get_old_and_generate(self, file_dir,town, scenario_key, ego_wp_index, ego_sp_index,other_wp_index_list):
        '''
        other_wp_index_list = [[other1wp,other1sp],..]
        '''
        self.scenario = self.load_scenario_dict(file_dir,town)
        
        if scenario_key not in self.scenario_dict:
            print(f"No scenario found with key {scenario_key}")
            return None
        scenario = self.scenario_dict[scenario_key]
        if ego_wp_index > len(scenario['wp']) - 1:
            print(f"Invalid ego wp index {ego_wp_index}")
            return None
        ego_wp = scenario['wp'][ego_wp_index]
        ego_sp =  scenario['wp'][ego_sp_index]
        ego_wp,ego_sp, ego_wp_direction = self.get_wp_sp_direction(ego_wp,del_sp=ego_sp)
        other_wp_sp_list = []
        for other_wp in other_wp_index_list():
            if len(other_wp[0]) > len(scenario['wp']) - 1:
                print(f"Invalid other wp index {other_wp['wp_id']}")
                continue
            other_wp =scenario['wp'][other_wp[0]]
            if other_wp == ego_wp:
                print(f"Other wp index {other_wp[0]} is the same as ego wp index")
                continue
            other_sp = scenario['wp'][other_sp[1]]
            other_wp_sp, other_wp_direction = self.get_wp_sp_direction(other_wp,del_sp=other_sp)
            other_wp_sp_list.append((other_wp, other_wp_sp, other_wp_direction))

        return scenario, ego_wp, ego_sp, ego_wp_direction, other_wp_sp_list

    def get_wp_sp_direction(self, wp, del_sp=None,direction=None, distance=None,choose_rev=False):
        # Find reachable waypoints and their directions
        reachable_wp = []
        for plan_way in self.scenario['plan_way']:
            dir_, length, path, reverse = plan_way
            if not choose_rev:
                reverse = True
            if wp in path:
                index = path.index(wp)
                if index < len(path) - 1 and reverse:
                    next_wp =path[-1]
                    reachable_wp.append((next_wp, dir_))
        if del_sp:
            reachable_wp = [(j,k) for (j,k) in reachable_wp if j !=del_sp]
        if len(reachable_wp) == 0:
            return None, None, None
        
        if distance is None:
            dis_ = S.MIN_DISTANCE
        else:
            dis_ = distance
            
        for i in range(100):
            if direction is None:
                sp, dic = random.choice(reachable_wp)
            else:
                reachable_d = [(w, d) for (w, d) in reachable_wp if d == S.TO_PLAN_DRICTION[direction]]
                if len(reachable_d) == 0:
                    sp, dic = random.choice(reachable_wp)
                else:
                    sp, dic = random.choice(reachable_d)
            distance_ = math.sqrt((wp[0]-sp[0])**2 + (wp[1]-sp[1])**2)
            if distance_ >= dis_:
                break
        return wp, sp, S.PLAN_DRICTION[dic]

    def get_random_ego_car_start_end_point(self,direction=None,distance=None):
        while True:
            wp = random.choice(self.scenario_sp)
            self.ego_car_wp, self.ego_car_sp, self.ego_car_directin = self.get_wp_sp_direction(wp=wp,direction=direction, distance=distance,choose_rev=True)
            if self.ego_car_wp:
                self.ego_car_wp_id = self.wp2node_index(self.ego_car_wp)
                self.ego_car_sp_id = self.wp2node_index(self.ego_car_sp)
                break

        return  self.ego_car_wp, self.ego_car_sp, self.ego_car_directin

    def wp2node_index(self,wp):
        
        return self.node_list.index(wp)

    @timeout(30) # 设置超时时间，单位为秒
    def get_random_other_car_start_end_point_with_timeout(self, ov_list_input):
        return self.get_random_other_car_start_end_point(ov_list_input)


    def get_wp_sp_direction_ov(self, wp):
        reachable_wp = []
        visited_wp = set()
        max_retries = 10  # 设定最大重试次数
        retry_count = 0

        while retry_count < max_retries:
            for plan_way in self.scenario['plan_way']:
                dir_, length, path, reverse = plan_way
                if wp in path:
                    index = path.index(wp)
                    if index < len(path) - 1:
                        for next_wp in path[index + 1:]:
                            next_wp_tuple = tuple(next_wp)  # 转换为元组
                            if next_wp_tuple not in visited_wp:
                                visited_wp.add(next_wp_tuple)
                                reachable_wp.append((next_wp_tuple, dir_))

            retry_count += 1

        if len(reachable_wp) == 0:
            return None, None, None
        sp, dic = random.choice(reachable_wp)
        return wp, sp, S.PLAN_DRICTION[dic]

    def get_random_other_car_start_end_point(self, ov_num):
        '''
        ov_list = [[dircetion,distance],...]
        '''
        ov_wp_sp_list = []
        other_wp = [w for w in self.scenario_sp if w != self.ego_car_wp]

        for index in range(ov_num):
            wp = other_wp[index]
            wp, sp, direction = self.get_wp_sp_direction_ov(wp=wp)
            if wp is None:
                continue
            if not isinstance(wp, list):
                wp = list(wp)
            if not isinstance(sp, list):
                sp = list(sp)
            ov_wp_sp_list.append([wp,sp, direction])

        if len(ov_wp_sp_list) != 0:
            for i, o in enumerate(ov_wp_sp_list):
                self.other_w_s_d_list[i] = {'wp': o[0], 'wp_id': self.wp2index(o[0]), 'sp': o[1], 'sp_id': self.wp2index(o[1]), 'direction': o[2]}

        return ov_wp_sp_list

    def add_scenario_to_graph(self,debug=False):
        '''
        node:pos
        edge:distance,way=0 egocar 1 other
        
        '''
        # Add nodes for all waypoints and traffic light locations
        index_i = 0
        for i, node_info in enumerate(self.scenario_node):
            node =node_info[0]
            is_in_wp = node_info[1]
            if is_in_wp is not None:
                try:            
                    node_mark_type=most_frequent_element(self.scenario['wp_mark'][is_in_wp])
                except:
                    node_mark_type = None
            else:
                node_mark_type = None
            node_mark_type = 'None' if node_mark_type is None or len(node_mark_type)==0 else node_mark_type
            self.G.add_node( i , pos=to_local_coordinate(self.ego_car_wp,node),type=S.WAYPOINT,mark_type = S.WP_MARK_DICT[node_mark_type],actor_name =S.DEFAULT)
            index_i +=1
        if len(self.scenario['tf_loc']) !=0:
            for i, node in enumerate(self.scenario['tf_loc']):                
                self.G.add_node(index_i, pos=to_local_coordinate(self.ego_car_wp,node),type=S.TRAFFIC,mark_type =S.WP_MARK_DICT['Signal_3Light_Post01'],actor_name =S.DEFAULT)
                index_i+=1
        # Add edges for all waypoints
        for i, node1 in enumerate(self.node_list):
            for j, node2 in enumerate(self.node_list):
                if i != j:
                    state,dirction = self.is_reachable(node1, node2)
                    if state:
                        weight = self.get_edge_weight(node1, node2)
                        self.G.add_edge(i, j, distance=weight,type=S.PLAN_DRICTION[dirction],way=S.NO_PLAN)


    def is_reachable(self, node1, node2):
        for plan_way in self.scenario['plan_way']:
            direction, length, path, reverse = plan_way
            if node1 in path and node2 in path:
                # index1 = path.index(node1)
                # index2 = path.index(node2)
                # if index2 - index1 == 1:
                return True,direction
                # elif index1 - index2 == 1 :
                    # return True,direction
        return False,None

    def get_edge_weight(self, node1, node2):
        return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2 + (node1[2] - node2[2]) **2)

    # def get_nearest_node(self, point):
    #     node_distances = [(i, np.linalg.norm(np.array(point) - np.array(node))) for i, node in enumerate(self.scenario_way_list)]
    #     return min(node_distances, key=lambda x: x[1])[0]

    def get_nearest_node(self, point):
        return self.wp2node_index(point)

    def G2tensor(self,weather_param):
        node_features = []
        edge_features = []
        edge_index = []
        for node in self.G.nodes():
            node_data = self.G.nodes[node]
            node_features.append([node_data['pos'][0], node_data['pos'][1], node_data['pos'][2],node_data['type'], node_data['mark_type'], node_data['actor_name']])
        
        for edge in self.G.edges():
            edge_data = self.G.edges[edge]
            edge_index.append([edge[0], edge[1]])
            edge_features.append([edge_data['distance'], edge_data['type'], edge_data['way']])

        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        weather_features= torch.tensor(weather_param, dtype=torch.float).unsqueeze(0)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,weather_attr=weather_features)
    
    def G_init(self):
        # 初始化G
        self.G = nx.Graph()            

    def G_save(self):
        nx.write_gpickle(self.G, "graph.pkl")
    
    def G2array(self):
        # 构建图中的节点和边属性
        node_pos = []
        node_type = []
        edge_index = []
        edge_distance = []
        edge_type = []
        node_mark_type=[]
        edge_way = []
        actor_name =[]
        for node in self.G.nodes():
            node_pos.append(list(self.G.nodes[node]['pos']))
            node_type.append(self.G.nodes[node]['type'])
            node_mark_type.append(self.G.nodes[node]['mark_type'])
            actor_name.append(self.G.nodes[node]['actor_name'])
        for edge in self.G.edges():
            edge_index.append(list(edge))
            edge_distance.append(self.G.edges[edge]['distance'])
            edge_type.append(self.G.edges[edge]['type'])
            edge_way.append(self.G.edges[edge]['way'])

        node_pos = np.array(node_pos)
        node_type = np.array(node_type)
        node_mark_type = np.array(node_mark_type)
        node_actor_name = np.array(actor_name)
        node_data = np.concatenate([node_pos, node_type.reshape(-1, 1), node_mark_type.reshape(-1, 1),node_actor_name.reshape(-1,1)], axis=1)
        # 将节点和边属性转换为张量
        x = torch.tensor( node_data, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([edge_distance, edge_type,edge_way], dtype=torch.float)

        # 创建PyTorch Geometric的Data对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.graph_data = data.to_dict()

    def to_python(self,obj): 
        if isinstance(obj, torch.Tensor): 
            return obj.tolist() 
        elif isinstance(obj, dict): 
            return {self.to_python(k): self.to_python(v) for k, v in obj.items()} 
        elif isinstance(obj, list): 
            return [self.to_python(item) for item in obj] 
        else: 
            return obj

    def save_json(self,filepath,score,error,value=None):
        
        self.G2array()
        scenario_data = {}
        scenario_data['town'] = self.town
        scenario_data['scenario_type'] = self.scearnio_type
        scenario_data['scenario_id'] = self.scearnio_id
        scenario_data['ego_car'] = {'wp':self.ego_car_wp,'wp_id':self.ego_car_wp_id,'sp':self.ego_car_sp,'sp_id':self.ego_car_sp_id,'direction':self.ego_car_directin}        
        scenario_data['other_vehicle'] = self.other_w_s_d_list
        scenario_data['data'] =self.to_python(self.graph_data)
        scenario_data['value'] =value
        scenario_data['score'] = score
        scenario_data['error'] = error
        
        with open(filepath, 'w') as f: 
            json.dump(scenario_data, f)


if __name__ == '__main__':
    town_list = ['Town01','Town02','Town03','Town04','Town05']
    file_path = 'scenario_lib'
    for town in town_list:
        ScenarioSelector.change_scearnio_file(file_path,town)
    # town = 'Town03'
    # sc = ScenarioSelector()
    # sc.load_scenario_dict('scenario_lib','Town03')
    # type_sc_id = random.choice(sc.scearnio_type_list)
    # scenario_type = sc.get_scenario_type(type_sc_id)
    # scenario_id  = random.choice([k for (k,v) in scenario_type])
    # sc.get_scenario(scenario_id)
    # w_,s_,d_ = sc.get_random_ego_car_start_end_point()
    # ov_list = [[None,None]]
    # ov = sc.get_random_other_car_start_end_point(ov_list)

    # sc.add_scenario_to_graph()
    
    # sc.save_json('test.json')
    

