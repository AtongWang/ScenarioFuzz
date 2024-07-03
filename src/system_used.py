import os,sys

config_path = os.path.abspath(__file__)
src_dir = os.path.dirname(config_path)
proj_root = os.path.dirname(src_dir)

try:
    sys.path.append(os.path.join(proj_root, "team_code"))
except IndexError:
     pass
sys.path.append('..')
from LAV.lav_agent import LAVAgent
from NEAT.neat_agent import MultiTaskAgent
from Transfuser.transfuser_agent import TransFuserAgent
from LAV_V2.lav_agent import LAVAgent as LAVAgentV2
from Transfuser_V2.submission_agent import HybridAgent as TransFuserAgentV2
L_SYSTEM_A={0:TransFuserAgent,1: MultiTaskAgent,2:LAVAgent,3:LAVAgentV2,4:TransFuserAgentV2}

L_SYSTEM_FILE = {0:'team_code/Transfuser/best_model.pth',
                       1:'team_code/NEAT/model_ckpt',
                       2:'team_code/LAV/config.yaml',
                       3:'team_code/LAV_V2/config.yaml',
                       4:'team_code/Transfuser_V2/model_ckpt/models_2022/transfuser'}
