import torch
import torch.nn as nn
from typing import Optional
from transformers.models.clip.modeling_clip import CLIPTextConfig, CLIPTextEmbeddings

def dig_to_str(tensor_list):
    return '_'.join([str(t.item()) for t in tensor_list])

#virtual token
class VirtualTokenManager(nn.Module):
    def __init__(self, categories=None, pretrained_embeddings=None):
        super(VirtualTokenManager, self).__init__()
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(49408, 1024).to('cuda')
        else:
            self.emb = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.emb.weight.requires_grad = False
        self.end=self.emb(torch.tensor([49407], dtype=torch.long).to('cuda'))
        self.zero = self.emb(torch.tensor([0], dtype=torch.long).to('cuda'))
        self.virtual_tokens = nn.ParameterDict()
        if categories is not None:
            if categories.dim() == 2:
                for category in categories:
                    tem_arr=[]
                    for i in category[1:]:
                        if i !=49407:
                            tem_arr.append(i)
                        else:
                            break
                    self.virtual_tokens[dig_to_str(tem_arr)] = nn.Parameter(self.emb(torch.as_tensor(tem_arr, dtype=torch.long).clone().to('cuda')))
            else:
                tem_arr = []
                for i in categories[1:]:
                    if i != 49407:
                        tem_arr.append(i)
                    else:
                        break
                self.virtual_tokens[dig_to_str(tem_arr)] = nn.Parameter(
                    self.emb(torch.as_tensor(tem_arr, dtype=torch.long).clone().to('cuda')))

    def load_from_state_dict(self, state_dict):
        self.emb.load_state_dict({'weight': state_dict['emb.weight']})
        self.end=self.emb(torch.tensor([49407], dtype=torch.long).to('cuda'))
        self.zero=self.emb(torch.tensor([0], dtype=torch.long).to('cuda'))
        for key in state_dict:
            if key.startswith('virtual_tokens'):
                token_key = key.split('.')[1]
                self.virtual_tokens[token_key] = nn.Parameter(state_dict[key])

    def forward(self, categories):
        batch_tokens = []
        for category in categories:
            tem_arr = []
            c_len = len(category)
            for i, j in enumerate(category):
                if j != 49407:
                    tem_arr.append(j)
                else:
                    break
            tem_tensor=self.virtual_tokens['_'.join([str(i) for i in tem_arr])]
            left=c_len-len(tem_tensor)
            tem_end=self.end
            if left >0:
                if left>1:
                    tem_end=self.zero.repeat(left+(len(tem_arr)-tem_tensor.size(0)), 1) if category[len(category)-left+1]==0 else self.end.repeat(left, 1)
                tem_end=torch.cat((self.end, tem_end), dim=0)
            batch_tokens.append(torch.cat((tem_tensor, tem_end.detach()), dim=0))
        return torch.stack(batch_tokens)





# combine prompt embedding and virtual token
class CustomCLIPTextEmbeddings(CLIPTextEmbeddings):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.virtual_tokens = None

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        if input_ids[0][1] !=49407:
            categories = input_ids[:, 5:-1].tolist()
            try:
                input_new= torch.cat([inputs_embeds[:, :5, :].detach(), self.virtual_tokens(categories)], dim=1)
            except RuntimeError as e:
                print(e)
        else:
            input_new = inputs_embeds



        position_embeddings = self.position_embedding(position_ids)
        try:
            embeddings = input_new + position_embeddings
        except RuntimeError as e:
            print(e)

        return embeddings

