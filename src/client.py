import transformers.models.clip.modeling_clip as clip_modeling
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import torch.nn as nn
import yaml
import torch.optim as optim
import copy
import json
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import diffusers
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, AutoProcessor, AutoTokenizer,CLIPTextConfig
from .CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPModel, CLIPVisionModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from .read_cifar_data import CustomImageDataset, CustomDataLoader, CustomSDDataLoader
from .read_combine_data import CustomCombineImageDataset, CustomCombineDataLoader
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image


class Client:
    def __init__(self, client_id, config_file, writer):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        clip_modeling.CLIPTextEmbeddings = CustomCLIPTextEmbeddings
        self.clip_model_path=config['model_path']['clip_model_path']
        self.sd_model_path=config['model_path']['sd_model_path']
        self.save_model_path=config['model_path']['save_model_path']

        self.prompt_model_lr=config['prompt_model']['lr']
        self.prompt_model_batch_size=config['prompt_model']['batch_size']
        self.prompt_model_weight_decay=config['prompt_model']['weight_decay']
        self.prompt_model_num_epochs=config['prompt_model']['num_epochs']
        self.prompt_model_save_freq=config['prompt_model']['save_freq']
        self.prompt_snr_gamma=config['prompt_model']['snr_gamma']

        self.num_inference_steps=config['inference_model']['num_inference_steps']
        self.guidance_scale=config['inference_model']['guidance_scale']
        self.image_num=config['inference_model']['image_num']
        self.save_image_path=config['inference_model']['save_image_path']

        self.dataset_type=config['data']['dataset_type']

        self.image_encoder_lr=config['image_encoder']['lr']
        self.image_encoder_batch_size=config['image_encoder']['batch_size']
        self.image_encoder_weight_decay=config['image_encoder']['weight_decay']
        self.image_encoder_num_epochs=config['image_encoder']['num_epochs']
        self.image_encoder_save_freq=config['image_encoder']['save_freq']
        self.lora_r=config['image_encoder']['lora_r']
        self.lora_alpha=config['image_encoder']['lora_alpha']
        self.lora_dropout=config['image_encoder']['lora_dropout']
        self.target_modules=config['image_encoder']['target_modules']
        self.generated_images_weights=config['image_encoder']['generated_images_weights']

        self.test_result_address=config['test_result']['test_result_address']
        self.seed=config['client']['seed']

        if self.dataset_type == 'Cifar10':
            self.type_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.k=10
        elif self.dataset_type=='Cifar100':
            self.type_list = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
                'mountain',
                'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
                'pine_tree',
                'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                'tiger', 'tractor',
                'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]
            self.k=100
        else:
            with open('src/Tinyimagenet.json', 'r', encoding='utf-8') as file:
                self.type_list= json.load(file)
            self.k=200

        # clip and stable-diffusion model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clipmodel = CLIPModel.from_pretrained(self.clip_model_path)
        cliptokenizer = CLIPTokenizer.from_pretrained(self.clip_model_path)

        #dataset
        self.client_id=client_id
        train_data_dir = os.path.join('dataset', self.dataset_type, 'train', str(self.client_id)+'.npz')
        test_data_dir = os.path.join('dataset', self.dataset_type, 'test', str(self.client_id)+'.npz')
        self.train_image, self.train_label=self.load_data(train_data_dir, 'train')

        def numeric_sort(file_name):
            return int(file_name.split('_')[1].split('.')[0])

        if self.dataset_type=='Tiny-imagenet':
            self.test_image_all = []
            val_data_dir=os.path.join('dataset', self.dataset_type, 'rawdata', 'tiny-imagenet-200', 'val', 'images')
            for root, dirs, files in os.walk(val_data_dir):
                for file in sorted(files, key=numeric_sort):
                    file_path = os.path.join(root, file)
                    self.test_image_all.append(file_path)
            with open('src/Tinylist.json', 'r', encoding='utf-8') as file:
                label_dict= json.load(file)
            val_labal_dir=os.path.join('dataset', self.dataset_type, 'rawdata', 'tiny-imagenet-200', 'val', 'val_annotations.txt')
            self.test_label_all = []
            with open(val_labal_dir, 'r') as file:
                for line in file:
                    parts = line.split()
                    category_id = parts[1]
                    self.test_label_all.append(label_dict[category_id])
        else:
            all_test_data_dir = os.path.join('dataset', self.dataset_type, 'test', 'all.npz')
            self.test_image_all, self.test_label_all = self.load_data(all_test_data_dir, 'test')


        self.test_image, self.test_label=self.load_data(test_data_dir, 'test')

        self.writer=writer

        #prompt
        self.train_prompt=list(set(self.train_label))
        self.text_inputs = cliptokenizer(self.train_prompt, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        self.total_inputs= cliptokenizer(self.type_list, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        self.pretrained_embeddings = clipmodel.text_model.embeddings.token_embedding.weight.to(self.device)
        self.vt = VirtualTokenManager(self.total_inputs, self.pretrained_embeddings).to(self.device)


        self.message_type={}
        for i, name in enumerate(self.type_list):
            self.message_type[name]=i

        #the chained prompt at this round
        self.prompt_index=0
        self.round=-1





    def load_data(self, address, type):
        with open(address, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        x_train = torch.Tensor(train_data['x']).type(torch.float32).to('cpu')
        y_train = torch.Tensor(train_data['y']).type(torch.int64).to('cpu')
        to_pil = transforms.ToPILImage()
        x_train = (x_train + 1) / 2
        images = [to_pil(x_train[i]) for i in range(len(x_train))]
        data_dir = os.path.join('dataset', self.dataset_type, type, str(self.client_id))
        image_paths=[]
        for i, image in enumerate(images):
            image_path=os.path.join(data_dir, f'image_{i}.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            image_paths.append(image_path)
        label=[self.type_list[i.item()] for i in y_train]
        return image_paths, label

# set requires_grad as false except the prompt which will be trained at this round
    def set_requires_grad(self, grad_keys):
        for name, param in self.vt.virtual_tokens.items():
            if name in grad_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def prompt_train(self):
        clipmodel = CLIPModel.from_pretrained(self.clip_model_path).to(self.device)
        clipprocessor = AutoProcessor.from_pretrained(self.clip_model_path)

        for param in clipmodel.parameters():
            param.requires_grad = False

        clipmodel.text_model.embeddings.virtual_tokens = self.vt
        self.prompt_index = (self.prompt_index + 1)%len(self.train_prompt)
        self.round +=1
        grad_index=self.text_inputs[self.prompt_index]
        tem_arr = []
        for i in grad_index[1:]:
            if i != 49407:
                tem_arr.append(i)
            else:
                break
        grad_keys='_'.join([str(t.item()) for t in tem_arr])
        self.set_requires_grad(grad_keys)
        optimizer = optim.AdamW(self.vt.virtual_tokens.parameters(), lr=self.prompt_model_lr, weight_decay=self.prompt_model_weight_decay)


        train_dataset = CustomImageDataset(self.train_image, self.train_label,filter_labels=self.train_prompt[self.prompt_index])

        total_label = [f'a photo of a {label}' for label in self.train_prompt]
        dataloader = CustomDataLoader(train_dataset, batch_size=self.prompt_model_batch_size, shuffle=True, preprocess=clipprocessor, total_labels=total_label, device=self.device)
        for epoch in range(self.prompt_model_num_epochs):
            for return_value in dataloader:
                logit = clipmodel(**return_value)
                labels=torch.tensor([self.prompt_index]*logit.logits_per_image.size(0)).to(device=self.device)
                loss = F.cross_entropy(logit.logits_per_image, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
                # self.writer.add_scalar(f'Client_{self.client_id}/round_{self.round}/Prompt_Training_Loss', loss.item(), epoch)
            print(f"Epoch[{epoch}/{self.prompt_model_num_epochs}], Loss: {loss.item():.4f}")
            if (epoch + 1) % self.prompt_model_save_freq == 0:
                tem_save_path=os.path.join(self.save_model_path, self.dataset_type,str(self.client_id), str(self.prompt_index), self.train_prompt[self.prompt_index], f"checkpoint_{epoch+1}.pth")
                os.makedirs(os.path.dirname(tem_save_path), exist_ok=True)
                torch.save(self.vt.state_dict(), tem_save_path)
        self.emb_message=nn.Parameter(self.vt.virtual_tokens[grad_keys].clone(), requires_grad=False)

        del clipmodel
        torch.cuda.empty_cache()

    def sd_prompt_train(self):
        vae = AutoencoderKL.from_pretrained(self.sd_model_path, subfolder="vae").to(self.device)
        vae.requires_grad_(False)

        tokenizer = CLIPTokenizer.from_pretrained(self.sd_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.sd_model_path, subfolder="text_encoder").to(self.device)
        text_encoder.requires_grad_(False)

        unet = UNet2DConditionModel.from_pretrained(self.sd_model_path, subfolder="unet").to(self.device)
        unet.requires_grad_(False)

        scheduler = PNDMScheduler.from_pretrained(self.sd_model_path, subfolder="scheduler")

        text_encoder.text_model.embeddings.virtual_tokens = self.vt
        self.prompt_index = (self.prompt_index + 1) % len(self.train_prompt)
        self.round += 1
        grad_index = self.text_inputs[self.prompt_index]
        tem_arr = []
        for i in grad_index[1:]:
            if i != 49407:
                tem_arr.append(i)
            else:
                break
        grad_keys = '_'.join([str(t.item()) for t in tem_arr])
        self.set_requires_grad(grad_keys)
        optimizer = optim.AdamW(self.vt.virtual_tokens.parameters(), lr=self.prompt_model_lr,
                                weight_decay=self.prompt_model_weight_decay)

        train_dataset = CustomImageDataset(self.train_image, self.train_label,
                                           filter_labels=self.train_prompt[self.prompt_index])
        dataloader = CustomSDDataLoader(train_dataset, batch_size=self.prompt_model_batch_size, shuffle=True,
                                      preprocess=tokenizer, device=self.device)
        for epoch in range(self.prompt_model_num_epochs):
            for labels, images in dataloader:
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=self.device)
                timesteps = timesteps.long()

                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(labels.input_ids, return_dict=False)[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                alphas_cumprod = scheduler.alphas_cumprod.to(self.device)  # α_t 表
                alpha_t = alphas_cumprod[timesteps]
                snr = alpha_t / (1 - alpha_t)
                mse_loss_weights = torch.stack([snr, self.prompt_snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                mse_loss_weights = mse_loss_weights / snr

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
            print(f"Epoch[{epoch}/{self.prompt_model_num_epochs}], Loss: {loss.item():.4f}")
            if (epoch + 1) % self.prompt_model_save_freq == 0:
                tem_save_path=os.path.join(self.save_model_path, self.dataset_type,str(self.client_id), str(self.prompt_index), self.train_prompt[self.prompt_index], f"checkpoint_{epoch+1}.pth")
                os.makedirs(os.path.dirname(tem_save_path), exist_ok=True)
                torch.save(self.vt.state_dict(), tem_save_path)
        self.emb_message=nn.Parameter(self.vt.virtual_tokens[grad_keys].clone(), requires_grad=False)
        del vae, text_encoder, unet
        torch.cuda.empty_cache()



    def collect_message(self,messages, client_id, round, train_index):
        cliptokenizer = CLIPTokenizer.from_pretrained(self.clip_model_path)
        emb=messages.clone()
        emb=emb.to(self.device)

        tem_index = f"{client_id}_{round}"
        text_inputs = cliptokenizer([tem_index], padding=True, return_tensors="pt")
        tem_arr = []
        for i in text_inputs['input_ids'][0][1:]:
            if i != 49407:
                tem_arr.append(i)
            else:
                break
        tt='_'.join([str(t.item()) for t in tem_arr])
        self.message_type[tem_index]=train_index+self.k
        self.vt.virtual_tokens[tt] = emb

    def inference(self, this_round_message):
        pipeline = StableDiffusionPipeline.from_pretrained(self.sd_model_path).to(self.device)
        pipeline.text_encoder.text_model.embeddings.virtual_tokens = self.vt
        images_pair=[]
        for i in this_round_message:
            prompt=[f'a photo of a {i}']
            for j in range(self.image_num):
                generator = torch.manual_seed(self.seed+j)
                result = pipeline(prompt, guidance_scale=self.guidance_scale, num_inference_steps=self.num_inference_steps,
                      generator=generator)
                tem_save_path = os.path.join(self.save_image_path, str(self.client_id), str(self.round)+"_round",i, f"{j}.png")
                t_image=result.images[0]
                os.makedirs(os.path.dirname(tem_save_path), exist_ok=True)
                t_image.save(tem_save_path)
                images_pair.append((i, tem_save_path))
        del pipeline
        torch.cuda.empty_cache()
        return images_pair

    def cluster(self):
        tem_dict=self.vt.virtual_tokens
        numpy_array=[value.detach().cpu().numpy() for value in tem_dict.values()]
        max_length = max([value.shape[0] for value in numpy_array])
        padded_array = [np.pad(value, ((0, max_length - value.shape[0]), (0, 0)), mode='constant') for value in
                        numpy_array]
        num=np.stack(padded_array, axis=0)
        num_flattened = num.reshape(num.shape[0], -1)
        init_centers = num_flattened[:self.k]
        kmeans = KMeans(n_clusters=self.k, init=init_centers, n_init=1, max_iter=1)
        labels = kmeans.fit_predict(num_flattened)
        for key, label in zip(self.message_type.keys(), labels):
            self.message_type[key]=label


    def image_encoder_train(self):
        self.cluster()
        clipmodel = CLIPModel.from_pretrained(self.clip_model_path).to(self.device)
        clipprocessor = AutoProcessor.from_pretrained(self.clip_model_path)
        clip_vision_model = CLIPVisionModel.from_pretrained(self.clip_model_path).to(self.device)
        clipmodel.text_model.embeddings.virtual_tokens = self.vt
        for param in clipmodel.parameters():
            param.requires_grad = False
        if self.round==0:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
            )
            new_vision_model = get_peft_model(clip_vision_model, lora_config)
        else:
            tem_save_path = os.path.join(self.save_model_path, "image_encoder", self.dataset_type,
                                         str(self.client_id) + "_client_id")
            new_vision_model = PeftModel.from_pretrained(clip_vision_model, tem_save_path)
            for name, param in new_vision_model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            new_vision_model.train()
        clipmodel.vision_model=new_vision_model
        optimizer = optim.AdamW(new_vision_model.parameters(), lr=self.image_encoder_lr, weight_decay=self.image_encoder_weight_decay)
        tem_total_labels = copy.deepcopy(self.type_list)
        for label in self.this_round_message:
            tem_total_labels.append(label)

        image_dataset=CustomCombineImageDataset(self.train_image, self.train_label, self.generated_messages, self.generated_images_weights)
        image_dataloader = CustomCombineDataLoader(image_dataset, batch_size=self.image_encoder_batch_size, shuffle=True,
                                      preprocess=clipprocessor, total_labels=tem_total_labels,message_type=self.message_type, device=self.device)
        print(f'{self.client_id}_{self.round} image encoder train start')
        for epoch in range(self.image_encoder_num_epochs):
            for return_value, labels, weights in image_dataloader:
                logit = clipmodel(**return_value)
                loss = F.cross_entropy(logit.logits_per_image, labels)
                weights_loss=(loss * weights).mean()
                optimizer.zero_grad()
                weights_loss.backward()  # 清空上一步的梯度 # 计算梯度
                optimizer.step()
                print(loss.item())
            # self.writer.add_scalar(f'Client_{self.client_id}/round_{self.round}/image_encoderTraining_Loss', loss.item(), epoch)
            print(f"Epoch[{epoch}/{self.prompt_model_num_epochs}], Loss: {loss.item():.4f}")
        tem_save_path = os.path.join(self.save_model_path, "image_encoder", self.dataset_type,
                                     str(self.client_id) + "_client_id")
        os.makedirs(os.path.dirname(tem_save_path), exist_ok=True)
        new_vision_model.save_pretrained(tem_save_path)
        del clipmodel, clip_vision_model
        torch.cuda.empty_cache()


    def exchange_message_and_generate(self, other_clients):

        self.this_round_message=[]
        for i, neighbor in enumerate(other_clients):
            self.collect_message(neighbor.emb_message, neighbor.client_id, neighbor.round, i)
            tem_index=f'{neighbor.client_id}_{neighbor.round}'
            self.this_round_message.append(tem_index)

        self.generated_messages = self.inference(self.this_round_message)

    def model_test(self, is_trained=False):
        clipmodel = CLIPModel.from_pretrained(self.clip_model_path).to(self.device)
        clipmodel.text_model.embeddings.virtual_tokens = self.vt
        clipprocessor = AutoProcessor.from_pretrained(self.clip_model_path)
        if is_trained:
            clip_vision_model = CLIPVisionModel.from_pretrained(self.clip_model_path).to(self.device)
            tem_save_path = os.path.join(self.save_model_path, "image_encoder", self.dataset_type,
                                         str(self.client_id) + "_client_id")
            new_vision_model = PeftModel.from_pretrained(clip_vision_model, tem_save_path)
            clipmodel.vision_model = new_vision_model
        clipmodel.eval()
        image_dataset = CustomCombineImageDataset(self.test_image, self.test_label, None)
        image_dataloader = CustomCombineDataLoader(image_dataset, batch_size=self.image_encoder_batch_size,
                                                   shuffle=True,
                                                   preprocess=clipprocessor, total_labels=self.type_list,
                                                   message_type=self.message_type, device=self.device)
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for return_value, labels, _ in image_dataloader:
                logit = clipmodel(**return_value)
                predictions = logit.logits_per_image.argmax(dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += len(labels)

        accuracy = correct_predictions / total_samples
        if os.path.exists(self.test_result_address):
            df = pd.read_csv(self.test_result_address)
        else:
            df = pd.DataFrame(columns=["client_id", "round", "accuracy"])

        mask = (df["client_id"] == self.client_id) & (df["round"] == self.round)

        if mask.any():
            df.loc[mask, "accuracy"] = accuracy
        else:
            new_row = {"client_id": self.client_id, "round": self.round, "accuracy": accuracy}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.writer.add_scalar(f'Client_{self.client_id}/accuracy', accuracy, self.round)
        df.to_csv(self.test_result_address, index=False)
        del clipmodel
        torch.cuda.empty_cache()

    def model_test_all(self, is_trained=False):
        clipmodel = CLIPModel.from_pretrained(self.clip_model_path).to(self.device)
        clipmodel.text_model.embeddings.virtual_tokens = self.vt
        clipprocessor = AutoProcessor.from_pretrained(self.clip_model_path)
        if is_trained:
            clip_vision_model = CLIPVisionModel.from_pretrained(self.clip_model_path).to(self.device)
            tem_save_path = os.path.join(self.save_model_path, "image_encoder", self.dataset_type,
                                         str(self.client_id) + "_client_id")
            new_vision_model = PeftModel.from_pretrained(clip_vision_model, tem_save_path)
            clipmodel.vision_model = new_vision_model
        clipmodel.eval()
        image_dataset = CustomCombineImageDataset(self.test_image_all, self.test_label_all, None)
        image_dataloader = CustomCombineDataLoader(image_dataset, batch_size=self.image_encoder_batch_size,
                                                   shuffle=True,
                                                   preprocess=clipprocessor, total_labels=self.type_list,
                                                   message_type=self.message_type, device=self.device)
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for return_value, labels, _ in image_dataloader:
                logit = clipmodel(**return_value)
                predictions = logit.logits_per_image.argmax(dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += len(labels)

        accuracy = correct_predictions / total_samples
        if os.path.exists(self.test_result_address):
            df = pd.read_csv(self.test_result_address)
        else:
            df = pd.DataFrame(columns=["client_id", "round", "accuracy"])

        mask = (df["client_id"] == self.client_id) & (df["round"] == self.round)

        if mask.any():
            df.loc[mask, "accuracy"] = accuracy
        else:
            new_row = {"client_id": self.client_id, "round": self.round, "accuracy": accuracy}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.writer.add_scalar(f'Client_{self.client_id}/all_accuracy', accuracy, self.round)
        df.to_csv(self.test_result_address, index=False)
        del clipmodel
        torch.cuda.empty_cache()



















