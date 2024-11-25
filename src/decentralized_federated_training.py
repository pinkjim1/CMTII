import yaml
import random
def decentralized_federated_learning(clients, config_file):

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    num_rounds=config['prompt_model']['round']
    p=config['client']['p']

    num_selected_clients = int(len(clients) * p)
    new_clients = random.sample(clients, num_selected_clients)

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")

        # local train
        for client in new_clients:
            if round ==0 and client.client_id==0:
                client.model_test(is_trained=False)
            client.prompt_train()

        # fd train
        for i, client in enumerate(new_clients):
            client.exchange_message_and_generate([c for c in clients if c != client])
            client.image_encoder_train()
            if round%2==0 and i<6:
                client.model_test(is_trained=True)

        print("Models exchanged and aggregated.")



