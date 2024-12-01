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
        for i, client in enumerate(new_clients):
            if round ==0 and i==0:
                client.model_test(is_trained=False)
                client.model_test_all(is_trained=False)
            client.prompt_train()

        # fd train
        for i, client in enumerate(new_clients):
            client.exchange_message_and_generate([c for c in new_clients if c != client])
            client.image_encoder_train()
            if round%2==0 and i<6:
                client.model_test(is_trained=True)
                client.model_test_all(is_trained=True)

        print("Models exchanged and aggregated.")



