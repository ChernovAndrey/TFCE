import numpy as np
import pandas as pd
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData, helper_load, helper_load_train
from tqdm import tqdm

from .base.evaluator import ProxyEvaluator
from .base.utils import *


class TFCEMLP_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total=len(self.data.train_loader))
        for batch_i, batch in pbar:
            batch = [x.to(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, mask = batch[0], batch[1], batch[2], batch[3], \
                batch[6]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]

            self.model.train()

            loss = self.model(users, pos_items, neg_items, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            num_batches += 1

        return [running_loss / num_batches]


class TFCEMLP_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)

    def add_special_model_attr(self, args):
        self.lm_model = args.lm_model
        loading_path = args.data_path + args.dataset + '/item_info/'
        embedding_path_dict = {
            'bert': 'item_cf_embeds_bert_array.npy',
            'roberta': 'item_cf_embeds_roberta_array.npy',
            'v2': 'item_cf_embeds_array.npy',
            'v3': 'item_cf_embeds_large3_array.npy',
            'v3_shuffle': "item_cf_embeds_large3_array_shuffle.npy",
            'llama2_7b': 'item_cf_embeds_llama2_7b_array.npy',
            'llama3_7b': 'item_cf_embeds_llama3_7b_instruct_array.npy',
            'mistral_7b': 'item_cf_embeds_Norm_Mistral-7B-v0.1_array.npy',
            'SFR': 'item_cf_embeds_Norm_SFR-Embedding-Mistral_7b_array.npy',
            'GritLM_7b': 'item_cf_embeds_Norm_GritLM-7B_array.npy',
            'e5_7b': 'item_cf_embeds_Norm_e5-mistral-7b-instruct_array.npy',
            'echo_7b': 'item_cf_embeds_Norm_echo-mistral-7b-instruct-lasttoken_array.npy',
        }
        self.item_cf_embeds = np.load(loading_path + embedding_path_dict[self.lm_model])


        def group_agg(group_data, embedding_dict, key='item_id'):
            ids = group_data[key].values
            embeds = [embedding_dict[id] for id in ids]
            embeds = np.array(embeds)
            return embeds.mean(axis=0)

        pairs = []
        for u, v in self.train_user_list.items():
            for i in v:
                pairs.append((u, i))
        pairs = pd.DataFrame(pairs, columns=['user_id', 'item_id'])

        # User CF Embedding: the average of item embeddings
        groups = pairs.groupby('user_id')
        item_cf_embeds_dict = {i: self.item_cf_embeds[i] for i in range(len(self.item_cf_embeds))}
        user_cf_embeds = groups.apply(group_agg, embedding_dict=item_cf_embeds_dict, key='item_id')
        user_cf_embeds_dict = user_cf_embeds.to_dict()
        user_cf_embeds_dict = dict(sorted(user_cf_embeds_dict.items(), key=lambda item: item[0]))
        self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values()))  # TODO: random init embeddings


class TFCEMLP(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.model_version = args.model_version
        self.is_batch_ensemble = args.is_batch_ensemble
        self.n_ensemble_members = args.n_ensemble_members if self.is_batch_ensemble else 1

        self.init_item_cf_embeds = data.item_cf_embeds
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).to(self.device)
        self.init_embed_shape = self.init_item_cf_embeds.shape[1]

        self.init_user_cf_embeds = data.user_cf_embeds
        self.init_user_cf_embeds = torch.tensor(self.init_user_cf_embeds, dtype=torch.float32).to(self.device)



        self.set_graph_embeddings()
        # To keep the same parameter size
        multiplier_dict = {
            'bert': 8,
            'roberta': 8,
            'v2': 2,
            'v3': 1 / 2,
            'v3_shuffle': 1 / 2,
        }
        if (self.lm_model in multiplier_dict):
            multiplier = multiplier_dict[self.lm_model]
        else:
            multiplier = 9 / 32  # for dimension = 4096

        if self.is_batch_ensemble:
            # BatchEnsemble implementation with shared weights and rank-1 adaptation (Wen et al., ICLR 2020)
            if (self.model_version == 'homo'):  # Linear mapping
                # Shared weight matrix
                self.mlp_shared_weight = nn.Parameter(torch.randn(self.embed_size, self.init_embed_shape))
                self.mlp_user_shared_weight = nn.Parameter(torch.randn(self.embed_size, self.init_embed_shape))
                
                # Rank-1 adaptation vectors r_i and s_i for each ensemble member
                self.mlp_r_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.embed_size))
                self.mlp_s_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.init_embed_shape))
                self.mlp_user_r_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.embed_size))
                self.mlp_user_s_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.init_embed_shape))
                
                # Initialize shared weights
                nn.init.kaiming_uniform_(self.mlp_shared_weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.mlp_user_shared_weight, a=math.sqrt(5))
                
                # Initialize rank-1 vectors to have small variance
                nn.init.normal_(self.mlp_r_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_s_vectors, mean=1.0, std=0.1) 
                nn.init.normal_(self.mlp_user_r_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_user_s_vectors, mean=1.0, std=0.1)
            else:  # MLP
                # Shared weight matrices for both layers
                hidden_dim = int(multiplier * self.init_embed_shape)
                
                # First layer shared weights and rank-1 vectors
                self.mlp_layer1_shared_weight = nn.Parameter(torch.randn(hidden_dim, self.init_embed_shape))
                self.mlp_layer1_r_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, hidden_dim))
                self.mlp_layer1_s_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.init_embed_shape))
                
                self.mlp_user_layer1_shared_weight = nn.Parameter(torch.randn(hidden_dim, self.init_embed_shape))
                self.mlp_user_layer1_r_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, hidden_dim))
                self.mlp_user_layer1_s_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.init_embed_shape))
                
                # Second layer shared weights and rank-1 vectors
                self.mlp_layer2_shared_weight = nn.Parameter(torch.randn(self.embed_size, hidden_dim))
                self.mlp_layer2_r_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.embed_size))
                self.mlp_layer2_s_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, hidden_dim))
                
                self.mlp_user_layer2_shared_weight = nn.Parameter(torch.randn(self.embed_size, hidden_dim))
                self.mlp_user_layer2_r_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, self.embed_size))
                self.mlp_user_layer2_s_vectors = nn.Parameter(torch.randn(self.n_ensemble_members, hidden_dim))
                
                # Initialize shared weights using standard initialization
                nn.init.kaiming_uniform_(self.mlp_layer1_shared_weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.mlp_layer2_shared_weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.mlp_user_layer1_shared_weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.mlp_user_layer2_shared_weight, a=math.sqrt(5))
                
                # Initialize rank-1 vectors to have small variance around 1.0
                nn.init.normal_(self.mlp_layer1_r_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_layer1_s_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_user_layer1_r_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_user_layer1_s_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_layer2_r_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_layer2_s_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_user_layer2_r_vectors, mean=1.0, std=0.1)
                nn.init.normal_(self.mlp_user_layer2_s_vectors, mean=1.0, std=0.1)
        else:
            # Original implementation
            if (self.model_version == 'homo'):  # Linear mapping
                self.mlp = nn.Sequential(
                    nn.Linear(self.init_embed_shape, self.embed_size, bias=False)  # homo
                )

                self.mlp_user = nn.Sequential(
                    nn.Linear(self.init_embed_shape, self.embed_size, bias=False)  # homo
                )
            else:  # MLP

                self.mlp = nn.Sequential(
                    nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
                    nn.LeakyReLU(),
                    nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
                )

                self.mlp_user = nn.Sequential(
                    nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
                    nn.LeakyReLU(),
                    nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
                )

    def init_embedding(self):
        pass

    def set_graph_embeddings(self):
        print('applying GCN to LLM embeddings')

        all_emb = torch.cat([self.init_user_cf_embeds, self.init_item_cf_embeds])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        self.init_user_cf_embeds, self.init_item_cf_embeds = torch.split(light_out,
                                                                         [self.data.n_users, self.data.n_items])

    def compute(self):
        if self.is_batch_ensemble:
            if self.model_version == 'homo':
                # BatchEnsemble with rank-1 adaptation for homo version
                user_outputs = []
                item_outputs = []
                
                for ensemble_idx in range(self.n_ensemble_members):
                    # Create ensemble-specific weight matrix: W_i = W ⊙ (r_i @ s_i^T)
                    # where ⊙ is Hadamard product
                    r_i = self.mlp_r_vectors[ensemble_idx:ensemble_idx+1]  # (1, embed_size)
                    s_i = self.mlp_s_vectors[ensemble_idx:ensemble_idx+1]  # (1, init_embed_shape)
                    rank_one_matrix = torch.mm(r_i.T, s_i)  # (embed_size, init_embed_shape)
                    ensemble_weight = self.mlp_shared_weight * rank_one_matrix
                    
                    user_r_i = self.mlp_user_r_vectors[ensemble_idx:ensemble_idx+1]
                    user_s_i = self.mlp_user_s_vectors[ensemble_idx:ensemble_idx+1]
                    user_rank_one_matrix = torch.mm(user_r_i.T, user_s_i)
                    user_ensemble_weight = self.mlp_user_shared_weight * user_rank_one_matrix
                    
                    # Apply ensemble-specific transformations
                    item_out = F.linear(self.init_item_cf_embeds, ensemble_weight)
                    user_out = F.linear(self.init_user_cf_embeds, user_ensemble_weight)
                    
                    user_outputs.append(user_out)
                    item_outputs.append(item_out)
                
                # Average ensemble predictions
                users = torch.stack(user_outputs).mean(dim=0)
                items = torch.stack(item_outputs).mean(dim=0)
            else:
                # BatchEnsemble with rank-1 adaptation for MLP version
                user_outputs = []
                item_outputs = []
                
                for ensemble_idx in range(self.n_ensemble_members):
                    # First layer
                    r1_i = self.mlp_layer1_r_vectors[ensemble_idx:ensemble_idx+1]
                    s1_i = self.mlp_layer1_s_vectors[ensemble_idx:ensemble_idx+1]
                    rank_one_matrix1 = torch.mm(r1_i.T, s1_i)
                    ensemble_weight1 = self.mlp_layer1_shared_weight * rank_one_matrix1
                    
                    user_r1_i = self.mlp_user_layer1_r_vectors[ensemble_idx:ensemble_idx+1]
                    user_s1_i = self.mlp_user_layer1_s_vectors[ensemble_idx:ensemble_idx+1]
                    user_rank_one_matrix1 = torch.mm(user_r1_i.T, user_s1_i)
                    user_ensemble_weight1 = self.mlp_user_layer1_shared_weight * user_rank_one_matrix1
                    
                    # Apply first layer
                    item_hidden = F.leaky_relu(F.linear(self.init_item_cf_embeds, ensemble_weight1))
                    user_hidden = F.leaky_relu(F.linear(self.init_user_cf_embeds, user_ensemble_weight1))
                    
                    # Second layer
                    r2_i = self.mlp_layer2_r_vectors[ensemble_idx:ensemble_idx+1]
                    s2_i = self.mlp_layer2_s_vectors[ensemble_idx:ensemble_idx+1]
                    rank_one_matrix2 = torch.mm(r2_i.T, s2_i)
                    ensemble_weight2 = self.mlp_layer2_shared_weight * rank_one_matrix2
                    
                    user_r2_i = self.mlp_user_layer2_r_vectors[ensemble_idx:ensemble_idx+1]
                    user_s2_i = self.mlp_user_layer2_s_vectors[ensemble_idx:ensemble_idx+1]
                    user_rank_one_matrix2 = torch.mm(user_r2_i.T, user_s2_i)
                    user_ensemble_weight2 = self.mlp_user_layer2_shared_weight * user_rank_one_matrix2
                    
                    # Apply second layer
                    item_out = F.linear(item_hidden, ensemble_weight2)
                    user_out = F.linear(user_hidden, user_ensemble_weight2)
                    
                    user_outputs.append(user_out)
                    item_outputs.append(item_out)
                
                # Average ensemble predictions
                users = torch.stack(user_outputs).mean(dim=0)
                items = torch.stack(item_outputs).mean(dim=0)
        else:
            # Original implementation
            users = self.mlp_user(self.init_user_cf_embeds)
            items = self.mlp(self.init_item_cf_embeds)

        return users, items

    def forward(self, users, pos_items, neg_items, mask):

        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if not self.data.is_one_pos_item:
            return supcon_loss(users_emb, pos_emb, neg_emb, mask, self.tau, 0)

        if (self.train_norm):
            users_emb = F.normalize(users_emb, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)
            neg_emb = F.normalize(neg_emb, dim=-1)

        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1))

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim=2)

        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))
        return ssm_loss

    # @torch.no_grad()
    @torch.inference_mode()
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).to(self.device)]
        items = all_items[torch.tensor(items).to(self.device)]

        if (self.pred_norm == True):
            users = F.normalize(users, dim=-1)
            items = F.normalize(items, dim=-1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items)  # user * item

        return rate_batch.cpu().detach().numpy()
