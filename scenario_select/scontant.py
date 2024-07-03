LANE_TYPE={'None':-1,'staright':0,'cross_intersection':1,'t_intersection':2,'X_intersection':3, 'Y_intersection':4,'5_intersection':5}
WP_MARK_DICT={ 'default':-1,'None':0,'Signal_3Light_Post01':1,  'Speed_60':2, 'Speed_30':3, 'Stencil_STOP':4,'Speed_90':5,  'Sign_Stop':6, 'Sign_Yield':7, 'speed_30':8}
WP_LANE_TYPE ={'default':-1,'NONE':0,'Right':1,'Left':-1,'Both':2,'LeftOnRed':3,'RightOnRed':4}
PLAN_DRICTION={'Unknown':0,'Left':-1,'Right':1,'Straight':2}

TO_LANE_TYPE = {-1:'None',0: 'staright', 1: 'cross_intersection', 2: 't_intersection', 3: 'X_intersection', 4: 'Y_intersection', 5: '5_intersection'} 
TO_WP_MARK_DICT = {-1:'default',0:'None',1: 'Signal_3Light_Post01', 2: 'Speed_60', 3: 'Speed_30', 4: 'Stencil_STOP', 5: 'Speed_90', 6: 'Sign_Stop', 7: 'Sign_Yield', 8: 'speed_30'} 
TO_WP_LANE_TYPE = {-1:'default',0: 'NONE', 1: 'Right', -1: 'Left', 2: 'Both', 3: 'LeftOnRed', 4: 'RightOnRed'} 
TO_PLAN_DRICTION = {0: 'Unknown', -1: 'Left', 1: 'Right', 2: 'Straight'}
# NODE TYPE
WAYPOINT =0
TRAFFIC =1
EGO_CAR_SP =2
EGO_CAR_WP = 3
OTHER_VEHICLE_SP =4
OTHER_VEHICLE_WP = 5 
OTHER_WALKER_SP = 6
OTHER_WALKER_WP = 7 

#ACTOR_TYPE
DEFAULT = -1

#SPEED
DEFAULT_SPEED = -1

NO_PLAN=0
EGO_EDGE = 1
OV_EDGE= 2
OW_EDGE= 3



MIN_DISTANCE =15