import sys
sys.path.append('carla/PythonAPI/carla/dist/carla-0.9.10-py3.6-linux-x86_64.egg')
import carla

# 获取Carla客户端
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 获取Carla世界
world = client.get_world()
blueprint_library = world.get_blueprint_library()
# 获取所有的walker和vehicle对象
actors = blueprint_library.filter('walker.p*') 
# + world.get_actors().filter('vehicle*')

# 建立名称索引
actor_dict = {}
for actor in actors:
    actor_name = actor.id
    id =int(actor_name[-4:])
    actor_dict[id] = actor_name
actor_dict = {k:actor_dict[k] for k in range(1,len(actor_dict)+1)}
print(actor_dict)
# 打印名称索引


# 获取可用地图
world_map = client.get_available_maps()

# 打印所有可用地图
print('Available maps:')
for map_name in world_map:
    print(map_name)