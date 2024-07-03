import os,sys
import torch    
sys.path.append('src/scenario_eval_model')
from eval_model import GNNBinaryClassificationModel, FocalLoss,ImprovedGNNBinaryClassificationModel






def get_model(model_name, device=None):
    model = None
    model_path = None
    if 'basic' in model_name:
        node_input_dim = 6
        edge_input_dim = 3
        weather_input_dim = 8
        hidden_dim = 64
        model = GNNBinaryClassificationModel(node_input_dim, edge_input_dim, weather_input_dim, hidden_dim)
        version = model_name.split('-')[-1] if 'v' in model_name else 'v0'
        model_path = f'src/scenario_eval_model/model_dir/basic_{version}/gnn_binary_classification_model_0.pt'
    elif  'improve' in model_name:
        node_input_dim = 6
        edge_input_dim = 3
        weather_input_dim = 8
        hidden_dim = 64
        model = ImprovedGNNBinaryClassificationModel(node_input_dim, edge_input_dim, weather_input_dim, hidden_dim, num_heads=4)
        version = model_name.split('-')[-1] if 'v' in model_name else 'v0'
        model_path = f'src/scenario_eval_model/model_dir/improve_{version}/gnn_binary_classification_model_1.pt'

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    if model is not None and model_path is not None:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
    else:
        raise ValueError("Model or model_path is not defined.")
    return model