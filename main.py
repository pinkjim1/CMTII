import torch
import numpy as np
import random
import yaml
from src.CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from src.decentralized_federated_training import decentralized_federated_learning
from src.client import Client
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


config_file="configs/config.yaml"



def create_clients(num_clients, config_file, writer):
    clients = []
    for client_id in range(num_clients):
        clients.append(Client(client_id, config_file, writer))
    return clients


with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# create clients
num_clients = config['client']['num']
seed=config['client']['seed']
writer_add=config['test_result']['writer']

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

writer = SummaryWriter(writer_add)

clients = create_clients(num_clients, config_file, writer)


clients = decentralized_federated_learning(clients, config_file)
