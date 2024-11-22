import torch
import torch.nn as nn
import yaml
from p_tuning.CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from p_tuning.decentralized_federated_training import decentralized_federated_learning
from p_tuning.test import test_federated_learning
from p_tuning.client import Client
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs/federated_learning8')

config_file="p_tuning/config.yaml"

def create_clients(num_clients, config_file, writer):
    clients = []
    for client_id in range(num_clients):
        clients.append(Client(client_id, config_file, writer))
    return clients


with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# create clients
num_clients = config['client']['num']
clients = create_clients(num_clients, config_file, writer)


clients = decentralized_federated_learning(clients, config_file)
