import yaml
def decentralized_federated_learning(clients, config_file):
    """
    去中心化联邦学习的主要流程。
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    num_rounds=config['prompt_model']['round']

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")

        # local train
        for client in clients:
            if round ==0 and client.client_id==0:
                client.model_test(is_trained=False)
            client.prompt_train()

        # fd train
        for client in clients:
            client.exchange_message_and_generate([c for c in clients if c != client])
            client.image_encoder_train()
            if round%2==0 and client.client_id<6:
                client.model_test(is_trained=True)

        print("Models exchanged and aggregated.")



