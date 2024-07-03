import os,sys
import random 
sys.path.append('src')
import config 
config.set_carla_api_path()
import copy
try:
    import carla
except ModuleNotFoundError as e:
    print("[-] Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)
import math
import networkx as nx
import argparse
import json,yaml
import numpy as np
import itertools


def convert_set_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_set_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_set_to_list(elem) for elem in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, tuple):
        return [convert_set_to_list(elem) for elem in obj]
    else:
        return obj



# 首先定义一个函数，用于获取node对应的waypoint的属性
def get_node_wp_properties(node):
    n_wp = node2wp(node, map)
    lane_id = n_wp.lane_id
    road_id = n_wp.road_id
    section_id = n_wp.section_id
    return (lane_id, road_id, section_id)

def node_cluster(node_list,dis,debug=False):
    global map
    global visited
    # 获取所有节点之间的距离矩阵
    dist_matrix = np.zeros((len(node_list), len(node_list)))
    for i, node_i in enumerate(node_list):
        for j, node_j in enumerate(node_list):
            dist_matrix[i][j] = node2wp(node_i, map).transform.location.distance(node2wp(node_j, map).transform.location)

    # 将距离矩阵中小于阈值的元素置为1，大于阈值的元素置为0
    dist_threshold = dis  # 可以根据实际情况调整阈值
    dist_matrix[dist_matrix < dist_threshold] = 1
    dist_matrix[dist_matrix >= dist_threshold] = 0

    # 使用深度优先搜索对节点进行分类
    visited = np.zeros(len(node_list))
    node_classes = []

    def dfs(node_idx, node_class):       
        visited[node_idx] = 1
        node_class.append(node_idx)
        for i, connected in enumerate(dist_matrix[node_idx]):
            if connected == 1 and visited[i] == 0:
                dfs(i, node_class)

    for i in range(len(node_list)):
        if visited[i] == 0:
            new_node_class = []
            dfs(i, new_node_class)
            node_classes.append(new_node_class)

    if debug:
        color_list = generate_random_colors(len(node_classes))        
        for i, node_class in enumerate(node_classes):
            draw_waypoints([node2wp(node_list[i],map) for i in node_class ],color_list[i])       
            print("Node class", i+1)
            for node_idx in node_class:
                node = node_list[node_idx]
                print("Node", node, "properties", get_node_wp_properties(node))
    return node_classes

def list_with_list(list1,list2):
    if set(list1).intersection(set(list2)) == set(list1):
        return True
    else:
        return False

def center_cal(waypoints):
    x_sum = y_sum = z_sum = 0
    a = len(waypoints)
    for w in waypoints:
        x_sum += w[0]
        y_sum += w[1]
        z_sum += w[2]
    center_location = (x_sum / a, y_sum / a, z_sum / a)
    return center_location


def type2str(i):
    if i == carla.libcarla.LaneChange.NONE:
        return 'straight'
    elif i == carla.libcarla.LaneChange.Right:
        return 'right'
    elif i == carla.libcarla.LaneChange.Left:
        return 'left'
    elif i == carla.libcarla.LaneChange.Both:
        return 'both'
    elif i == carla.libcarla.LaneChange.RightOnRed:
        return 'right on red'
    elif i == carla.libcarla.LaneChange.LeftOnRed:
        return 'left on red'
    else:
        return ''
def road_type_plt(scenario_dict_tf,road_type):
    global world
    '''
    scearnrio_dict
    road_type  all/5/cross/X/Y/T/.../None
    '''
    intersection_l = ['X','Y','T','cross','5']
    color_list = generate_random_colors(len(scenario_dict_tf))
        
    for i,v in scenario_dict_tf.items():
        if road_type != 'all':
            if road_type == 'None':
                if v['type'] is None:
                    scene_wp= [node2wp(j ,map) for j in v['wp']]
                    draw_waypoints(scene_wp,color_list[i])
            elif road_type in intersection_l:
                if v['type'] ==f'{road_type}_intersection':
                    scene_wp= [node2wp(j ,map) for j in v['wp']]
                    draw_waypoints(scene_wp,color_list[i])
            else:
                pass
        else:
            scene_wp= [node2wp(j ,map) for j in v['wp']]        
            draw_waypoints(scene_wp,color_list[i])
            str_t = (v['type'] if v['type'] is not None else 'None') + int(len(v['tf_loc'])!=0)*'_traffic'
            if v['center_loc'] is not None:
                world.debug.draw_string(carla.Location(*v['center_loc']), str_t, draw_shadow=False,
                                        color=carla.Color(*(255,0,0)), life_time=120.0,
                                        persistent_lines=True)

def generate_random_colors(num_colors):
    colors = []

    for i in range(num_colors):
        # 生成随机颜色
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # 将颜色加入列表
        colors.append((r, g, b))

    return colors

def type2int(i):
    if i == carla.libcarla.LaneChange.NONE:
        return 0
    elif i == carla.libcarla.LaneChange.Right:
        return 1
    elif i == carla.libcarla.LaneChange.Left:
        return -1
    elif i == carla.libcarla.LaneChange.Both:
        return 2
    elif i == carla.libcarla.LaneChange.RightOnRed:
        return 4
    elif i == carla.libcarla.LaneChange.LeftOnRed:
        return 3
    else:
        return ''



def draw_waypoints(waypoints, color):
    global world
    for wp in waypoints:
        world.debug.draw_string(wp.transform.location, 'O', draw_shadow=False,
                                color=carla.Color(r=color[0], g=color[1], b=color[2]), life_time=120.0,
                                persistent_lines=True)
        # world.debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(r=color[0], g=color[1], b=color[2]), life_time=120.0,
        #                         persistent_lines=True)
def node2wp(node,map):
    return map.get_waypoint(location=carla.Location(node[0],node[1],node[2]))


# 在Carla中显示生成的路点
def display_waypoints(straight_points, cross_points, x_points, t_points, y_points, intersection_points):
    # 将所有路点合并到一个列表中
    all_points = straight_points + cross_points + x_points + t_points + y_points + intersection_points

    # 将路点按照车道类型（Driving或者Sidewalk）进行分类
    driving_points = [wp for wp in all_points if wp.lane_type == carla.LaneType.Driving]
    sidewalk_points = [wp for wp in all_points if wp.lane_type == carla.LaneType.Sidewalk]

    # 将不同类型的路点用不同颜色标注
    draw_waypoints(driving_points, color=[255, 0, 0]) # Driving类型的路点用红色标注
    draw_waypoints(sidewalk_points, color=[0, 255, 0]) # Sidewalk类型的路点用绿色标注

def search_graph_p_node(graph,id):
    k_list =[]
    for k,v in graph:
        if id in v:
            k_list.append(k)
    return k_list

def is_within_distance_threshold(wp1, wp2, threshold):
    """
    判断两个Waypoint之间的距离是否小于等于给定阈值。
    :param wp1: 第一个Waypoint
    :param wp2: 第二个Waypoint
    :param threshold: 阈值，单位为米
    :return: True，如果两个Waypoint之间的距离小于等于阈值；False，否则。
    """
    loc1 = wp1.transform.location
    loc2 = wp2.transform.location
    distance = math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)
    return distance <= threshold+2 and distance >=threshold-2

def is_within_next_previous_wp(wp1,wp2,distance):
    return wp1 in wp2.next(distance) or wp1 in wp2.previous(distance)
# def find_stargith_wp(graph,wp_list):
#     wp_list =  []
#     for i, p in enumerate(wp_list):
#         p_node_l = search_graph_p_node(graph,i)
#         if len(p_node_l) == 0:
#             pass
#         else:
#             if p[0] == carla.LaneType.Driving and not p[1] and len(graph[i]) == 2

def get_components(node_list,debug=False):
    global G
    components = nx.connected_components(G.subgraph(node_list))
    # 将连通分量转换为列表
    component_list = list(components)
    # 打印连通分量的数量
    
    if debug:
        print('*'*10+"Number of connected components:{}".format(len(component_list))+'*'*10)
        # 打印每个连通分量中包含的节点数量
        for i, component in enumerate(component_list):
            print("Component", i+1, "size:", len(component))
    return component_list

def get_components_2(node_list, debug=False):
    global G
    component_list = []
    for node1 in node_list:
        for node2 in node_list:
            if node1 == node2:
                continue
            if nx.has_path(G, node1, node2):
                # 如果node1和node2之间有路径，那么它们属于同一个连通分量
                    component_list.append(set([node1, node2]))
    
    if debug:
        print('*' * 10 + "Number of connected components:{}".format(len(component_list)) + '*' * 10)
        # 打印每个连通分量中包含的节点数量
        for i, component in enumerate(component_list):
            print("Component", i+1, "size:", len(component))
    return component_list

def get_components_5(node_list, debug=False):
    global G
    component_list = []
    for combination in itertools.combinations(node_list, 5):
        subgraph = G.subgraph(combination)
        if nx.is_connected(subgraph):
            component_list.append(set(combination))
    
    if debug:
        print('*' * 10 + "Number of connected components:{}".format(len(component_list)) + '*' * 10)
        # 打印每个连通分量中包含的节点数量
        for i, component in enumerate(component_list):
            print("Component", i+1, "size:", len(component))
    return component_list        

def get_route_direction(component):
    path = []
    for node in component:
        location = carla.Location(x=node[0], y=node[1], z=node[2])
        waypoint = map.get_waypoint(location)
        path.append(waypoint)
    # 获取路径的起点和终点
    start_wp = path[0]
    end_wp = path[-1]

    # 计算路径的方向向量
    start_loc = start_wp.transform.location
    end_loc = end_wp.transform.location
    direction = carla.Vector2D(end_loc.x - start_loc.x, end_loc.y - start_loc.y)

    # 计算路径的方向角度
    angle = math.degrees(math.atan2(direction.y, direction.x))
    distance = start_loc.distance(end_loc)
    # 根据路径的方向角度判断路径的方向
    if -45.0 <= angle <= 45.0:
        return "Straight",distance
    elif 45.0 < angle <= 135.0:
        return "Left",distance
    elif -135.0 <= angle < -45.0:
        return "Right",distance
    else:
        return "Unknown",distance


def parse_args():
    parser = argparse.ArgumentParser(description="Process traffic intersection data")
    parser.add_argument("--town", type=str, default="Town03", help="Name of the town")
    parser.add_argument("--debug", action="store_true", default=True,help="Enable debug mode")
    parser.add_argument("--output-format", type=str, default="json", help="Output file format")
    parser.add_argument("--output-path", type=str, default="./", help="Output file path")
    parser.add_argument("-u", "--sim-host", default="localhost", type=str,
    help="Hostname of Carla simulation server")
    parser.add_argument("-p", "--sim-port", default=2000, type=int,
    help="RPC port of Carla simulation server")
    return parser.parse_args()

def main(town):
    global map,G,world
    args = parse_args()
    args.town = town

    debug = args.debug
    # Connect to the Carla server
    client = carla.Client(args.sim_host, args.sim_port)
    client.set_timeout(10.0)
    if debug:
        print(client.get_available_maps())
    world = client.load_world(args.town)

    map = world.get_map() 
    topology = map.get_topology()

    # 创建空图
    G = nx.Graph()
    # 添加节点
    for waypoint_pair in topology:
        waypoint1, waypoint2 = waypoint_pair
        node1 = (waypoint1.transform.location.x, waypoint1.transform.location.y, waypoint1.transform.location.z)
        node2 = (waypoint2.transform.location.x, waypoint2.transform.location.y, waypoint2.transform.location.z)
        G.add_node(node1)
        G.add_node(node2)

    # 添加边
    for waypoint_pair in topology:
        waypoint1, waypoint2 = waypoint_pair
        node1 = (waypoint1.transform.location.x, waypoint1.transform.location.y, waypoint1.transform.location.z)
        node2 = (waypoint2.transform.location.x, waypoint2.transform.location.y, waypoint2.transform.location.z)
        distance = carla.Location(node1[0],node1[1],node1[2]).distance(
            carla.Location(node2[0],node2[1],node2[2]))
        G.add_edge(node1, node2, weight=distance)

    # 打印图的节点和边数
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())


    #------------------------------------------根据traffic light位置进行聚类，记录-------------------------------------------
    # Get all the traffic lights and their locations
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    traffic_light_locations = [tl.get_location() for tl in traffic_lights]
    traffic_light_locations_list = [(t.x,t.y,t.z) for t in traffic_light_locations]
    traffic_lights_classes_list = node_cluster(traffic_light_locations_list,35,debug=debug)
    scenario_dict_tf = {}


    for i,tf in enumerate(traffic_lights_classes_list):
            scenario_dict_tf[i] = {'type': None, 'wp': [],'wp_mark':[], 'wp_lanetype':[],'tf_loc': [],'center_loc':None,'plan_way':[]}
            scenario_dict_tf[i]['tf_loc'] = [traffic_light_locations_list[j]  for j in tf]

    for k, v in scenario_dict_tf.items():
        v['center_loc'] = center_cal([loc for loc in v['tf_loc']])



    #-----------------------------------------traffic的wp进行分类整理-----------------------------------------

    node_list =[node for node in G.nodes]
    node_list_wp = [node2wp(node,map) for node in node_list]
    node_h = np.zeros(len(node_list))
    print(f'waypoint catch: {int(np.sum(node_h))}/{len(node_list)}')
    for t in range(4):
        for i,v in scenario_dict_tf.items():
            for tf_loc in v['tf_loc']:
                tf_n_dis = [node_w.transform.location.distance(carla.Location(*tf_loc)) if node_h[j] == 0 else 9999 for j,node_w in enumerate(node_list_wp) ]
                wp_i = tf_n_dis.index(min(tf_n_dis))
                if wp_i > 50:
                    continue
                else:
                    n_wp =node_list_wp[wp_i]
                    n_wp_lane_type = type2int(n_wp.lane_change)
                    n_mark = n_wp.get_landmarks(1)
                    n_mark_name = [i.name for i in n_mark] if len(n_mark)   !=0 else []
                    v['wp'].append(node_list[wp_i])
                    v['wp_mark'].append(n_mark_name)
                    v['wp_lanetype'].append(n_wp_lane_type)
                    node_h[wp_i] =1

    for i,node in enumerate (node_list):
        if not node_h[i] ==1:
            n_wp =node_list_wp[i]
            n_wp_lane_type = type2int(n_wp.lane_change)
            n_mark = n_wp.get_landmarks(1)
            n_mark_name = [i.name for i in n_mark] if len(n_mark)   !=0 else []
            for j,v in scenario_dict_tf.items():
                c_loc = carla.Location(v['center_loc'][0],v['center_loc'][1],0)
                if debug:
                    draw_waypoints([node2wp(v['center_loc'],map)],(255,0,0))
                w_loc = carla.Location(n_wp.transform.location.x,n_wp.transform.location.y,0)
                if  w_loc.distance(c_loc)<30 :
                        v['wp'].append(node)
                        v['wp_mark'].append(n_mark_name)
                        v['wp_lanetype'].append(n_wp_lane_type)
                        node_h[i] =1
                        break

    print(f'waypoint catch: {int(np.sum(node_h))}/{len(node_list)}')

    #-----------------------------------------traffic外的wp进行分类整理-----------------------------------------
    node_others = [n for i,n in enumerate(node_list) if node_h[i] == 0]
    node_others_classes = node_cluster(node_others, 25, False)
    del_his = []
    node_other_last = copy.deepcopy(node_others_classes)
    drive_way=[]
    for index, n_o_class_nodes in enumerate(node_others_classes):
        n_o_class_node = [node_others[i] for i in n_o_class_nodes]
        compoents_list = get_components(n_o_class_node)
        drive_way_s = []
        for index_2, com in enumerate(compoents_list):
            if len(com) == 1:
                obj =next(iter(com))
                obj_index = node_others.index(obj)
                if obj_index in node_other_last[index]:
                    node_other_last[index].remove(obj_index)
                    break
            else:
                direction,lenth = get_route_direction(com)
                drive_way_s.append((direction,lenth,com))
        if not all([len_i == 1 for len_i in [len(i) for i in compoents_list]]):
            drive_way.append(drive_way_s)
        else:
            del_his.append(index)

    node_other_last_ = [item for i, item in enumerate(node_other_last) if i not in del_his]

    id_sc_ = len(scenario_dict_tf)
    for index,n_o_last in enumerate(node_other_last_):
        v =  {'type': None, 'wp': [],'wp_mark':[], 'wp_lanetype':[],'tf_loc': [],'center_loc':None,'plan_way':[]}
        n_o_wp_loc = [node_others[id_] for id_ in n_o_last]
        for  i_ in  [node_list.index(j_) for j_ in n_o_wp_loc]:
            node_h[i_]  =1  
        v['wp'] = n_o_wp_loc
        n_o_wp = [node2wp(n_o,map) for n_o in n_o_wp_loc]
        v['wp_lanetype'] = [ type2int(n_o.lane_change) for n_o in n_o_wp]
        for n_o in n_o_wp:
            n_mark = n_o.get_landmarks(1)
            n_mark_name = [i.name for i in n_mark] if len(n_mark)   !=0 else []
            v['wp_mark'].append(n_mark_name)
        v['plan_way'] = drive_way[index]
        v['center_loc'] = center_cal([loc for loc in v['wp']])
        scenario_dict_tf[id_sc_+index] =v

    print(f'waypoint catch: {int(np.sum(node_h))}/{len(node_list)}')

    #---------------------------------对剩余节点进行处理----------------------------------------------

    node_others_last = [n for i,n in enumerate(node_list) if node_h[i] == 0]
    compoents_last = get_components(n_o_class_node,debug)
    drive_way_last =[]
    if all([len_i == 1 for len_i in [len(i) for i in compoents_last ]]):
        pass
    else:        
        for index_2, com in enumerate(compoents_last):
            if len(com) >1:
                direction,lenth = get_route_direction(com)
                drive_way_last.append((direction,lenth,com))

        v =  {'type': None, 'wp': [],'wp_mark':[], 'wp_lanetype':[],'tf_loc': [],'center_loc':None,'plan_way':[]}
        n_o_wp_loc = node_others_last 
        for  i_ in  [node_list.index(j_) for j_ in n_o_wp_loc]:
            node_h[i_]  =1  
        v['wp'] = n_o_wp_loc
        n_o_wp = [node2wp(n_o,map) for n_o in n_o_wp_loc]
        v['wp_lanetype'] = [ type2int(n_o.lane_change) for n_o in n_o_wp]
        for n_o in n_o_wp:
            n_mark = n_o.get_landmarks(1)
            n_mark_name = [i.name for i in n_mark] if len(n_mark)   !=0 else []
            v['wp_mark'].append(n_mark_name)
        v['plan_way'] = drive_way[index]
        v['center_loc'] = center_cal([loc for loc in v['wp']])
        scenario_dict_tf[id_sc_+1] =v
    print(f'waypoint catch: {int(np.sum(node_h))}/{len(node_list)}')
    #----------------------------------- 遍历图的每个节点,确定类型-----------------------------------------------------------

    # 先关注交通灯附近的区域。
    for node in G.nodes:
        # 获取该节点的所有邻居节点    
        n_wp = node2wp(node,map)
        n_wp_lane_type = type2int(n_wp.lane_change)
        neighbors = list(G.neighbors(node))
        neighbor_distances = [G[node][neighbor]["weight"] for neighbor in neighbors]
        nb_wp_l =[node2wp(i,map) for i in neighbors]
        nb_wp_lane_type =[type2int(wp.lane_change) for wp in nb_wp_l]
        n_mark = n_wp.get_landmarks(1)
        n_mark_name = [i.name for i in n_mark] if len(n_mark)   !=0 else []
        #print(n_mark_name)
        n_tf = False
        store_i = None

        for i,v in scenario_dict_tf.items():
            if node in v['wp']:
                store_i = i
                if len(v['tf_loc']) !=0:
                    n_tf =True
                break

        if n_tf:
            if scenario_dict_tf[store_i]['type'] is None or scenario_dict_tf[store_i]['type']=='staright' :
                if len(neighbor_distances) ==5 and max(neighbor_distances) <50:
                        scenario_dict_tf[store_i]['type'] = '5_intersection'           
                        print("Found 5-intersection intersection at node:", node)

                elif   len(neighbor_distances) == 4 and len(scenario_dict_tf[store_i]['tf_loc'])==4:
                    if max(neighbor_distances) >10: # max(neighbor_distances) > 30 and 
                        scenario_dict_tf[store_i]['type'] = 'cross_intersection'           
                        print("Found crossing at node:", node)           
                    else:
                        scenario_dict_tf[store_i]['type'] = 'X_intersection'           
                        print("Found X intersection at node:", node)

            # 如果存在三个邻居，则认为是一个T字路
                elif len(neighbor_distances) == 3:
                    if max(neighbor_distances) > 10:
                            scenario_dict_tf[store_i]['type'] = 't_intersection'           
                            print("Found T intersection at node:", node)
                    else:
                        scenario_dict_tf[store_i]['type'] = 'Y_intersection'     
                        print("Found Y intersection at node:", node)
                
                elif len(neighbor_distances) <=2:
                    pass
                else:
                    continue
            else:
                pass

        else:
            if store_i:
                if scenario_dict_tf[store_i]['type'] is None or scenario_dict_tf[store_i]['type']=='staright' :
                    if len(neighbor_distances) ==5 and max(neighbor_distances) <50:
                            scenario_dict_tf[store_i]['type'] = '5_intersection'           
                            print("Found 5-intersection intersection at node:", node)

                    elif   len(neighbor_distances) == 4 :
                        if max(neighbor_distances) >10: # max(neighbor_distances) > 30 and 
                            scenario_dict_tf[store_i]['type'] = 'cross_intersection'           
                            print("Found crossing at node:", node)           
                        else:
                            scenario_dict_tf[store_i]['type'] = 'X_intersection'           
                            print("Found X intersection at node:", node)

                # 如果存在三个邻居，则认为是一个T字路
                    elif len(neighbor_distances) == 3 :
                        if max(neighbor_distances) > 10:
                                scenario_dict_tf[store_i]['type'] = 't_intersection'           
                                print("Found T intersection at node:", node)
                        else:
                            scenario_dict_tf[store_i]['type'] = 'Y_intersection'     
                            print("Found Y intersection at node:", node)
                    
                    elif len(neighbor_distances) <= 2 :
                        scenario_dict_tf[store_i]['type'] = 'staright'           
                        print("Found staright way with traffic light at node:", node)
            else:
                continue

    for sc in scenario_dict_tf.values():
        if len(sc['wp'])<=30:
            compoents_list = get_components_2(sc['wp'],debug)
            for com_l in compoents_list:
                pl_d,lenth =get_route_direction(com_l)
                if (pl_d,lenth,com_l) not in sc['plan_way']:
                    sc['plan_way'].append((pl_d,lenth,com_l))



# -----------针对缺少直线道路的情况，由于前代码所有的思路均是以路口为主，故从全图搜索plan，为路径较长，且一直直行的wp，存入sceanriodict
    compoents_all_list = get_components(node_list)
    index_a = [len(i) for i in compoents_all_list]
    sorted_idx = [idx for idx, val in sorted(enumerate(index_a), key=lambda x: x[1], reverse=True)]
    add_i = 0
    add_i_n = len(scenario_dict_tf)
    for ind in range(len(sorted_idx)):
        pl_w = compoents_all_list[ind]
        pl_d,lenth =get_route_direction(pl_w)
        if pl_d =='Straight' and lenth>30:
            v= {'type': None, 'wp': [],'wp_mark':[], 'wp_lanetype':[],'tf_loc': [],'center_loc':None,'plan_way':[]}
            v['wp'] = [w for w in pl_w]
            v['type'] = 'straight'
            n_o_wp = [node2wp(n_o,map) for n_o in v['wp']]
            v['wp_lanetype'] = [ type2int(n_o.lane_change) for n_o in n_o_wp]
            for n_o in n_o_wp:
                n_mark = n_o.get_landmarks(1)
                n_mark_name = [i.name for i in n_mark] if len(n_mark)   !=0 else []
                v['wp_mark'].append(n_mark_name)
            v['plan_way'].append((pl_d,lenth,pl_w))
            v['center_loc'] = center_cal([loc for loc in v['wp']])
            scenario_dict_tf[add_i_n+add_i] =v
            add_i+=1
#----------------------------对交通点即剩余场景plan_way进行完善

    if args.output_format == 'json':
        file_path = os.path.join(args.output_path,f'{args.town}_scenario_lib.{args.output_format}')
                # 保存为json文件
        # 假设 scenario_dict_tf 是一个字典，其中包含 set 类型的值
        with open(file_path, 'w') as f_json:
            json.dump(convert_set_to_list(scenario_dict_tf), f_json)

    if args.output_format == 'yaml':
        file_path = os.path.join(args.output_path,f'{args.town}_scenario_lib.{args.output_format}')
        with open(file_path, 'w') as f_yaml:
            yaml.dump(convert_set_to_list(scenario_dict_tf), f_yaml, allow_unicode=True)
    
    if debug:
        road_type_plt(scenario_dict_tf,'all')




if __name__ == "__main__":
    for town in ['Town01','Town02','Town03','Town04','Town05','Town06']:
        main(town) 