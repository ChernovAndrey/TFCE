import random as rd
import collections
from types import new_class
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from parse import parse_args
import time
import torch
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from reckit import randint_choice
import os
import bisect


# Helper function used when loading data from files
def helper_load(filename):
    user_dict_list = {}
    item_dict = set()

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

    return user_dict_list, item_dict,


def helper_load_train(filename):
    user_dict_list = {}
    item_dict = set()
    item_dict_list = {}
    trainUser, trainItem = [], []

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            # print(line)
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            # LightGCN
            trainUser.extend([user] * len(items))
            trainItem.extend(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

            for item in items:
                if item in item_dict_list.keys():
                    item_dict_list[item].append(user)
                else:
                    item_dict_list[item] = [user]

    return user_dict_list, item_dict, item_dict_list, trainUser, trainItem


# It loads the data and creates a train_loader

class AbstractData:
    def __init__(self, args):

        self.user_neighbors = {}

        self.is_one_pos_item = getattr(args, 'is_one_pos_item', True)
        self.n_pos_samples =int(args.n_pos_samples) if hasattr(args, 'n_pos_samples') else 1
        self.path = args.data_path + args.dataset + '/cf_data/'
        self.train_file = self.path + 'train.txt'
        self.valid_file = self.path + 'valid.txt'
        self.test_file = self.path + 'test.txt'

        self.mix = True if 'mix' in args.dataset else False

        if (args.nodrop):
            self.train_nodrop_file = self.path + 'train_nodrop.txt'
        self.nodrop = args.nodrop

        self.candidate = args.candidate
        if (args.candidate):
            self.test_neg_file = self.path + 'test_neg.txt'
        self.batch_size = args.batch_size
        self.neg_sample = args.neg_sample
        self.device = torch.device(f"cuda:{args.cuda}" if args.cuda != -1 and torch.cuda.is_available() else "cpu")
        self.model_name = args.model_name

        self.user_pop_max = 0
        self.item_pop_max = 0
        self.infonce = args.infonce
        self.num_workers = args.num_workers
        self.dataset = args.dataset
        self.candidate = args.candidate

        # Number of total users and items
        self.n_users, self.n_items, self.n_observations = 0, 0, 0
        self.users = []
        self.items = []
        self.population_list = []
        self.weights = []

        # List of dictionaries of users and its observed items in corresponding dataset
        # {user1: [item1, item2, item3...], user2: [item1, item3, item4],...}
        # {item1: [user1, user2], item2: [user1, user3], ...}
        self.train_user_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        if (self.dataset == "tencent_synthetic" or self.dataset == "kuairec_ood"):
            self.test_ood_user_list_1 = collections.defaultdict(list)
            self.test_ood_user_list_2 = collections.defaultdict(list)
            self.test_ood_user_list_3 = collections.defaultdict(list)
        else:
            self.test_user_list = collections.defaultdict(list)

        # Used to track early stopping point
        self.best_valid_recall = -np.inf
        self.best_valid_epoch, self.patience = 0, 0

        self.train_item_list = collections.defaultdict(list)
        self.Graph = None
        self.trainUser, self.trainItem, self.UserItemNet = [], [], []
        self.n_interactions = 0
        if (self.dataset == "tencent_synthetic" or self.dataset == "kuairec_ood"):
            self.test_ood_item_list_1 = []
            self.test_ood_item_list_2 = []
            self.test_ood_item_list_3 = []
        else:
            self.test_item_list = []

        # Dataloader
        self.train_data = None
        self.train_loader = None

        self.load_data()
        # model-specific attributes
        self.add_special_model_attr(args)

        self.get_dataloader()

    def add_special_model_attr(self, args):
        pass

    # self.trainUser and self.trainItem are respectively the users and items in the training set, in the form of an interaction list.
    def load_data(self):
        # Load raw data
        raw_train_user_list, train_item, raw_train_item_list, raw_trainUser, raw_trainItem = helper_load_train(
            self.train_file)
        raw_valid_user_list, valid_item = helper_load(self.valid_file)

        raw_test_user_list, self.test_item_list = helper_load(self.test_file)

        if (self.nodrop):
            raw_train_nodrop_user_list, self.train_nodrop_item_list = helper_load(self.train_nodrop_file)

        if (self.candidate):
            raw_test_neg_user_list, self.test_neg_item_list = helper_load(self.test_neg_file)
        else:
            raw_test_neg_user_list, self.test_neg_item_list = None, None
        self.pop_dict_list = []

        # Filter users: only keep users who have training history
        users_with_train_history = set(raw_train_user_list.keys())
        print(f"Original users in train: {len(users_with_train_history)}")
        
        # Filter validation users to only include those with training history
        filtered_valid_user_list = {user: items for user, items in raw_valid_user_list.items() 
                                   if user in users_with_train_history}
        print(f"Filtered valid users: {len(filtered_valid_user_list)} (from {len(raw_valid_user_list)})")
        
        # Filter test users to only include those with training history
        filtered_test_user_list = {user: items for user, items in raw_test_user_list.items() 
                                  if user in users_with_train_history}
        print(f"Filtered test users: {len(filtered_test_user_list)} (from {len(raw_test_user_list)})")
        
        # Filter test negatives if they exist
        if raw_test_neg_user_list is not None:
            filtered_test_neg_user_list = {user: items for user, items in raw_test_neg_user_list.items() 
                                          if user in users_with_train_history}
            print(f"Filtered test neg users: {len(filtered_test_neg_user_list)} (from {len(raw_test_neg_user_list)})")
        else:
            filtered_test_neg_user_list = None
            
        # Filter nodrop users if they exist
        if (self.nodrop):
            filtered_train_nodrop_user_list = {user: items for user, items in raw_train_nodrop_user_list.items() 
                                              if user in users_with_train_history}
            print(f"Filtered nodrop users: {len(filtered_train_nodrop_user_list)} (from {len(raw_train_nodrop_user_list)})")
        
        # Create user ID mapping: sparse user IDs -> consecutive integers (0, 1, 2, ...)
        original_users = sorted(list(users_with_train_history))
        user_id_mapping = {original_id: new_id for new_id, original_id in enumerate(original_users)}
        reverse_user_mapping = {new_id: original_id for original_id, new_id in user_id_mapping.items()}
        
        print(f"User ID mapping created: {len(user_id_mapping)} users")
        print(f"Sample mapping: {dict(list(user_id_mapping.items())[:10])}")
        
        # Apply user ID remapping to all data structures
        self.train_user_list = collections.defaultdict(list)
        for original_user, items in raw_train_user_list.items():
            new_user = user_id_mapping[original_user]
            self.train_user_list[new_user] = items
            
        self.valid_user_list = collections.defaultdict(list)
        for original_user, items in filtered_valid_user_list.items():
            new_user = user_id_mapping[original_user]
            self.valid_user_list[new_user] = items
            
        self.test_user_list = collections.defaultdict(list)
        for original_user, items in filtered_test_user_list.items():
            new_user = user_id_mapping[original_user]
            self.test_user_list[new_user] = items
            
        if filtered_test_neg_user_list is not None:
            self.test_neg_user_list = collections.defaultdict(list)
            for original_user, items in filtered_test_neg_user_list.items():
                new_user = user_id_mapping[original_user]
                self.test_neg_user_list[new_user] = items
        else:
            self.test_neg_user_list = None
            
        if (self.nodrop):
            self.train_nodrop_user_list = collections.defaultdict(list)
            for original_user, items in filtered_train_nodrop_user_list.items():
                new_user = user_id_mapping[original_user]
                self.train_nodrop_user_list[new_user] = items

        # Update train_item_list with new user IDs
        self.train_item_list = collections.defaultdict(list)
        for original_item, original_users in raw_train_item_list.items():
            # Only include users that have training history
            filtered_users = [user_id_mapping[user] for user in original_users if user in user_id_mapping]
            if filtered_users:  # Only add if there are valid users
                self.train_item_list[original_item] = filtered_users
        
        # Update trainUser and trainItem lists with new user IDs
        self.trainUser = []
        self.trainItem = []
        for original_user, new_user in user_id_mapping.items():
            if original_user in raw_train_user_list:
                items = raw_train_user_list[original_user]
                self.trainUser.extend([new_user] * len(items))
                self.trainItem.extend(items)

        temp_lst = [train_item, valid_item, self.test_item_list]

        # Users are now consecutive integers from 0 to n_users-1
        self.users = list(range(len(user_id_mapping)))
        self.items = list(set().union(*temp_lst))
        self.items.sort()
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        print("n_users: ", self.n_users)
        print("n_items: ", self.n_items)

        # Now we can safely iterate through consecutive user IDs
        for i in range(self.n_users):
            self.n_observations += len(self.train_user_list[i])
            self.n_interactions += len(self.train_user_list[i])
            if i in self.valid_user_list.keys():
                self.n_interactions += len(self.valid_user_list[i])
            if (self.dataset == "tencent_synthetic" or self.dataset == "kuairec_ood"):
                if i in self.test_ood_user_list_1.keys():
                    self.n_interactions += len(self.test_ood_user_list_1[i])
                if i in self.test_ood_user_list_2.keys():
                    self.n_interactions += len(self.test_ood_user_list_2[i])
                if i in self.test_ood_user_list_3.keys():
                    self.n_interactions += len(self.test_ood_user_list_3[i])
            else:
                if i in self.test_user_list.keys():
                    self.n_interactions += len(self.test_user_list[i])
        print('average number observations per a user: ',
              1.0 * self.n_observations / self.n_users)  # TODO: calculate here percentiles and set up n_pos_samples as X% percentile
        
        # Store the mapping for potential future use
        self.user_id_mapping = user_id_mapping
        self.reverse_user_mapping = reverse_user_mapping
        # Population matrix
        pop_dict = {}
        for item, users in self.train_item_list.items():
            pop_dict[item] = len(users) + 1
        for item in range(0, self.n_items):
            if item not in pop_dict.keys():
                pop_dict[item] = 1

            self.population_list.append(pop_dict[item])

        pop_user = {key: len(value) for key, value in self.train_user_list.items()}
        pop_item = {key: len(value) for key, value in self.train_item_list.items()}
        self.pop_item = pop_item
        self.pop_user = pop_user
        # Convert to a unique value.
        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)

        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i

        self.user_pop_idx = np.zeros(self.n_users, dtype=int)
        self.item_pop_idx = np.zeros(self.n_items, dtype=int)
        # Convert the originally sparse popularity into dense popularity.
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            # print(key, value)
            self.item_pop_idx[key] = item_idx[value]

        user_pop_max = max(self.user_pop_idx)
        item_pop_max = max(self.item_pop_idx)

        self.user_pop_max = user_pop_max
        self.item_pop_max = item_pop_max

        self.sample_items = np.array(self.items, dtype=int)

        if (self.mix):
            self.add_mixed_data()
        else:
            self.selected_train, self.selected_valid, self.selected_test = [], [], []
            self.nu_info = []
            self.ni_info = []

    def add_mixed_data(self):
        self.selected_train, self.selected_valid, self.selected_test = [], [], []
        self.nu_info = []
        self.ni_info = []
        self.mixed_datasets = ['movie', 'book', 'game']
        # self.mixed_datasets = ['movie', 'book', 'game', 'electronic']
        for data_name in self.mixed_datasets:
            train_train_file_, valid_file_, test_file_ = self.path + 'train_' + data_name + '.txt', self.path + 'valid_' + data_name + '.txt', self.path + 'test_' + data_name + '.txt'
            train_user_list_, train_item_, __, ___, ____ = helper_load_train(train_train_file_)
            valid_user_list_, valid_item_ = helper_load(valid_file_)
            test_user_list_, test_item_ = helper_load(test_file_)

            temp_lst = [train_item_, valid_item_, test_item_]
            users_ = list(set(train_user_list_.keys()))
            items_ = list(set().union(*temp_lst))
            items_.sort()
            n_users_ = len(users_)
            n_items_ = len(items_)
            print(f"n_users_: {data_name}", n_users_)
            print(f"n_items_: {data_name}", n_items_)

            self.selected_train.append(train_user_list_)
            self.selected_valid.append(valid_user_list_)
            self.selected_test.append(test_user_list_)
            self.nu_info.append(n_users_)
            self.ni_info.append(n_items_)

            self.cum_ni_info = np.cumsum(self.ni_info)
            self.cum_ni_info = np.insert(self.cum_ni_info, 0, 0)
            self.cum_nu_info = np.cumsum(self.nu_info)
            self.cum_nu_info = np.insert(self.cum_nu_info, 0, 0)
        # self.train_file_movie, self.valid_file_movie, self.test_file_movie = self.path + 'train_movie.txt', self.path + 'valid_movie.txt', self.path + 'test_movie.txt'
        # self.train_user_list_movie, train_item_movie, __, ___, ____ = helper_load_train(self.train_file_movie)
        # self.valid_user_list_movie, valid_item_movie = helper_load(self.valid_file_movie)
        # self.test_user_list_movie, test_item_movie = helper_load(self.test_file_movie)

        # temp_lst = [train_item_movie, valid_item_movie, test_item_movie]
        # self.users_movie = list(set(self.train_user_list_movie.keys()))
        # self.items_movie = list(set().union(*temp_lst))
        # self.items_movie.sort()
        # self.n_users_movie = len(self.users_movie)
        # self.n_items_movie = len(self.items_movie)
        # print("n_users_movie: ", self.n_users_movie)
        # print("n_items_movie: ", self.n_items_movie)

        # self.train_file_book, self.valid_file_book, self.test_file_book = self.path + 'train_book.txt', self.path + 'valid_book.txt', self.path + 'test_book.txt'
        # self.train_user_list_book, train_item_book, __, ___, ____ = helper_load_train(self.train_file_book)
        # self.valid_user_list_book, valid_item_book = helper_load(self.valid_file_book)
        # self.test_user_list_book, test_item_book = helper_load(self.test_file_book)

        # temp_lst = [train_item_book, valid_item_book, test_item_book]
        # self.users_book = list(set(self.train_user_list_book.keys()))
        # self.items_book = list(set().union(*temp_lst))
        # self.items_book.sort()
        # self.n_users_book = len(self.users_book)
        # self.n_items_book = len(self.items_book)
        # print("n_users_book: ", self.n_users_book)
        # print("n_items_book: ", self.n_items_book)

        # self.train_file_game, self.valid_file_game, self.test_file_game = self.path + 'train_game.txt', self.path + 'valid_game.txt', self.path + 'test_game.txt'
        # self.train_user_list_game, train_item_game, __, ___, ____ = helper_load_train(self.train_file_game)
        # self.valid_user_list_game, valid_item_game = helper_load(self.valid_file_game)
        # self.test_user_list_game, test_item_game = helper_load(self.test_file_game)

        # temp_lst = [train_item_game, valid_item_game, test_item_game]
        # self.users_game = list(set(self.train_user_list_game.keys()))
        # self.items_game = list(set().union(*temp_lst))
        # self.items_game.sort()
        # self.n_users_game = len(self.users_game)
        # self.n_items_game = len(self.items_game)
        # print("n_users_game: ", self.n_users_game)
        # print("n_items_game: ", self.n_items_game)

        # self.exclude_items = [self.items_movie, self.items_book]
        # self.selected_train = [self.train_user_list_movie, self.train_user_list_book]
        # self.nui_info = [[self.n_users_movie, self.n_items_movie], [self.n_users_book, self.n_items_book]]

        # self.selected_train = [self.train_user_list_movie, self.train_user_list_book, self.train_user_list_game]
        # self.nui_info = [[self.n_users_movie, self.n_items_movie], [self.n_users_book, self.n_items_book], [self.n_users_game, self.n_items_game]]

    def get_dataloader(self):

        self.train_data = TrainDataset(self.model_name, self.users, self.train_user_list, self.user_pop_idx,
                                       self.item_pop_idx, \
                                       self.neg_sample, self.n_observations, self.n_items, self.sample_items,
                                       self.infonce, self.items, self.nu_info, self.ni_info, self.is_one_pos_item,
                                       n_pos_samples=self.n_pos_samples, user_neighbors=self.user_neighbors)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, drop_last=True)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):

        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("finish loading adjacency matrix")
                norm_adj = pre_adj_mat

                self.trainItem = np.array(self.trainItem)
                self.trainUser = np.array(self.trainUser)
                self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                              shape=(self.n_users, self.n_items))
            # If there is no preprocessed adjacency matrix, generate one.
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                self.trainItem = np.array(self.trainItem)
                self.trainUser = np.array(self.trainUser)
                self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                              shape=(self.n_users, self.n_items))
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.tocsr()
                sp.save_npz(self.path + '/adj_mat.npz', adj_mat)
                print("successfully saved adj_mat...")

                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)

        # self.user_neighbors.update(get_user_neighbors_and_union_items(self.UserItemNet))
        # print(f'maximum number of excluded items:{np.array([v[2] for v in self.user_neighbors.values()]).max()}')
        return self.Graph


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, model_name, users, train_user_list, user_pop_idx, item_pop_idx, neg_sample, \
                 n_observations, n_items, sample_items, infonce, items, nu_info=None, ni_info=None,
                 is_one_pos_item=True, n_pos_samples=15, user_neighbors=None):
        self.is_one_pos_item = is_one_pos_item  # whether sample only 1 item
        self.user_neighbors = user_neighbors
        if not self.is_one_pos_item:
            self.n_pos_samples = n_pos_samples
            print('multiple positive samples are applied')
            print(f'number of positive samples:{n_pos_samples}')
        self.model_name = model_name
        self.users = users
        self.train_user_list = train_user_list
        self.user_pop_idx = user_pop_idx
        self.item_pop_idx = item_pop_idx
        self.neg_sample = neg_sample
        self.n_observations = n_observations
        self.n_items = n_items
        self.sample_items = sample_items
        self.infonce = infonce
        self.items = items

        self.nu_info = nu_info
        self.ni_info = ni_info
        self.cum_ni_info = np.cumsum(self.ni_info)
        self.cum_ni_info = np.insert(self.cum_ni_info, 0, 0)
        self.cum_nu_info = np.cumsum(self.nu_info)
        self.cum_nu_info = np.insert(self.cum_nu_info, 0, 0)

    def __getitem__(self, index):
        index = index % len(self.users)
        user = self.users[index]
        if self.train_user_list[user] == []:
            pos_items = 0
            mask = 0
        else:
            if self.is_one_pos_item:
                pos_item = rd.choice(self.train_user_list[user])
                mask = 1
            else:
                pos_item = rd.sample(self.train_user_list[user], self.n_pos_samples) if \
                    len(self.train_user_list[user]) > self.n_pos_samples else \
                    self.train_user_list[user][:self.n_pos_samples]
                mask = torch.zeros(self.n_pos_samples).long()
                mask[:len(pos_item)] = 1
                if len(pos_item) < self.n_pos_samples:
                    pos_item += [-1] * (self.n_pos_samples - len(pos_item))
                pos_item = torch.tensor(pos_item).long()
        # print(pos_item)
        user_pop = self.user_pop_idx[user]
        if self.is_one_pos_item:
            pos_item_pop = self.item_pop_idx[pos_item]
        else:
            # TODO: it does not make a difference for alpharec, but for others will
            pos_item_pop = -1
        if self.infonce == 1 and self.neg_sample == -1:  # in-batch
            return user, pos_item, user_pop, pos_item_pop

        elif self.infonce == 1 and self.neg_sample != -1:  # InfoNCE negative sampling
            if (len(self.nu_info) > 0):
                assert False, 'should not be here'
                # period = index
                period = bisect.bisect_right(self.cum_nu_info, index) - 1
                # print(self.cum_ni_info)
                exclude_items = list(np.array(self.train_user_list[user]) - self.cum_ni_info[period])
                # print('perirod', period)
                # print("********************")
                # print('info', self.ni_info[period])

                # neg_items = [1]
                neg_items = randint_choice(self.ni_info[period], size=self.neg_sample, exclusion=exclude_items)
                neg_items = list(np.array(neg_items) + self.cum_ni_info[period])

                # if(index < self.nui_info[0][0]):
                #     neg_items = randint_choice(self.nui_info[0][1], size=self.neg_sample, exclusion=self.train_user_list[user])
                # elif(index < self.nui_info[0][0] + self.nui_info[1][0]):
                #     # neg_items = randint_choice(self.n_items, size=self.neg_sample, exclusion=self.train_user_list[user]+self.exclude_items[0])
                #     exclude_items = list(np.array(self.train_user_list[user]) - self.nui_info[0][1])
                #     neg_items = randint_choice(self.nui_info[1][1], size=self.neg_sample, exclusion=exclude_items)
                #     neg_items = list(np.array(neg_items) + self.nui_info[0][1])
                # else:
                #     exclude_items = list(np.array(self.train_user_list[user]) - self.nui_info[0][1] - self.nui_info[1][1])
                #     neg_items = randint_choice(self.nui_info[2][1], size=self.neg_sample, exclusion=exclude_items)
                #     neg_items = list(np.array(neg_items) + self.nui_info[0][1] + self.nui_info[1][1])

            else:
                # neg_items = randint_choice(self.n_items, size=self.neg_sample, exclusion=self.user_neighbors[user][1])
                neg_items = randint_choice(self.n_items, size=self.neg_sample, exclusion=self.train_user_list[user])
            neg_items_pop = self.item_pop_idx[neg_items]

            return user, pos_item, user_pop, pos_item_pop, torch.tensor(neg_items).long(), neg_items_pop, mask

        else:  # BPR negative sampling. (only sample one negative item)
            while True:
                idx = rd.randint(0, self.n_items - 1)
                neg_item = self.items[idx]

                if neg_item not in self.train_user_list[user]:
                    break

            neg_item_pop = self.item_pop_idx[neg_item]
            return user, pos_item, user_pop, pos_item_pop, neg_item, neg_item_pop

    def __len__(self):
        return self.n_observations


class MultiDatasetData(AbstractData):
    """
    Multi-dataset data class that handles training on multiple datasets simultaneously.
    Each dataset maintains its own interaction graph and negative sampling is isolated within datasets.
    """
    
    def __init__(self, args):
        self.multi_datasets = args.multi_datasets
        self.multi_datasets_path = args.multi_datasets_path if args.multi_datasets_path else args.data_path
        self.proportional_sampling = args.proportional_sampling
        self.equal_sampling = getattr(args, 'equal_sampling', False)
        self.dataset_sampling_weights = args.dataset_sampling_weights
        
        # Initialize dataset-specific information
        self.dataset_info = {}
        self.dataset_user_offsets = {}
        self.dataset_item_offsets = {}
        
        # Override dataset name and path for multi-dataset setup
        args.dataset = '_'.join(self.multi_datasets)  # Create combined dataset name
        self.path = os.path.join(self.multi_datasets_path, args.dataset, 'cf_data')
        os.makedirs(self.path, exist_ok=True)  # Ensure directory exists
        
        # Call parent constructor but override some methods
        super().__init__(args)

    def add_special_model_attr(self, args):
        """Override to handle model-specific attributes for multi-dataset setup"""
        self.lm_model = args.lm_model
        
        # Initialize combined embeddings list
        all_item_cf_embeds = []
        
        # Load embeddings for each dataset
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
        
        for dataset_name in self.multi_datasets:
            loading_path = os.path.join(self.multi_datasets_path, dataset_name, 'item_info')
            embed_file = os.path.join(loading_path, embedding_path_dict[self.lm_model])
            dataset_embeds = np.load(embed_file)
            all_item_cf_embeds.append(dataset_embeds)
        
        # Combine embeddings
        self.item_cf_embeds = np.concatenate(all_item_cf_embeds, axis=0)
        
        # Calculate user embeddings as average of their item embeddings
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
        self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values()))

    def load_data(self):
        """Load data from multiple datasets and combine them"""
        print(f"Loading multi-dataset data from: {self.multi_datasets}")
        
        # Initialize combined data structures
        self.train_user_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        self.test_user_list = collections.defaultdict(list)
        self.test_neg_user_list = collections.defaultdict(list)  # Initialize test negatives
        self.test_neg_item_list = []  # Initialize test negative items
        
        all_users = set()
        all_items = set()
        user_offset = 0
        item_offset = 0
        
        self.trainUser = []
        self.trainItem = []
        
        for dataset_idx, dataset_name in enumerate(self.multi_datasets):
            print(f"Processing dataset {dataset_idx}: {dataset_name}")
            
            # Load individual dataset
            dataset_path = os.path.join(self.multi_datasets_path, dataset_name, 'cf_data')
            train_file = os.path.join(dataset_path, 'train.txt')
            valid_file = os.path.join(dataset_path, 'valid.txt')
            test_file = os.path.join(dataset_path, 'test.txt')
            test_neg_file = os.path.join(dataset_path, 'test_neg.txt')
            
            # Load data for this dataset
            raw_train_user_list, train_item_set, raw_train_item_list, trainUser, trainItem = helper_load_train(train_file)
            raw_valid_user_list, valid_item_set = helper_load(valid_file)
            raw_test_user_list, test_item_set = helper_load(test_file)
            
            # Load test negatives if they exist
            if os.path.exists(test_neg_file):
                raw_test_neg_user_list, test_neg_item_list = helper_load(test_neg_file)
            else:
                raw_test_neg_user_list, test_neg_item_list = {}, []
            
            # Filter users: only keep users who have training history
            users_with_train_history = set(raw_train_user_list.keys())
            print(f"  Dataset {dataset_name}: Original users in train: {len(users_with_train_history)}")
            
            # Filter validation, test, and test negatives to only include users with training history
            filtered_valid_user_list = {user: items for user, items in raw_valid_user_list.items() 
                                       if user in users_with_train_history}
            filtered_test_user_list = {user: items for user, items in raw_test_user_list.items() 
                                      if user in users_with_train_history}
            filtered_test_neg_user_list = {user: items for user, items in raw_test_neg_user_list.items() 
                                          if user in users_with_train_history}
            
            print(f"  Dataset {dataset_name}: Filtered valid users: {len(filtered_valid_user_list)} (from {len(raw_valid_user_list)})")
            print(f"  Dataset {dataset_name}: Filtered test users: {len(filtered_test_user_list)} (from {len(raw_test_user_list)})")
            print(f"  Dataset {dataset_name}: Filtered test neg users: {len(filtered_test_neg_user_list)} (from {len(raw_test_neg_user_list)})")
            
            # Get unique users and items for this dataset (only users with training history)
            dataset_users = users_with_train_history
            dataset_items = set().union(train_item_set, valid_item_set, test_item_set)
            
            # Store dataset information
            self.dataset_info[dataset_name] = {
                'users': sorted(list(dataset_users)),
                'items': sorted(list(dataset_items)),
                'n_users': len(dataset_users),
                'n_items': len(dataset_items),
                'n_interactions': sum(len(items) for items in raw_train_user_list.values()),
                'user_offset': user_offset,
                'item_offset': item_offset,
                'train_user_list': raw_train_user_list,
                'valid_user_list': filtered_valid_user_list,
                'test_user_list': filtered_test_user_list,
                'test_neg_user_list': filtered_test_neg_user_list,
                'path': dataset_path  # Store the original dataset path
            }
            
            print(f"  Dataset {dataset_name}: {len(dataset_users)} users, {len(dataset_items)} items")
            
            # Combine users and items with offsets to ensure no conflicts
            for user in dataset_users:
                adjusted_user = user + user_offset
                all_users.add(adjusted_user)
                
                # Map original user interactions to adjusted IDs
                if user in raw_train_user_list:
                    self.train_user_list[adjusted_user] = [item + item_offset for item in raw_train_user_list[user]]
                    # Add to global trainUser/trainItem lists
                    self.trainUser.extend([adjusted_user] * len(raw_train_user_list[user]))
                    self.trainItem.extend([item + item_offset for item in raw_train_user_list[user]])
                    
                if user in filtered_valid_user_list:
                    self.valid_user_list[adjusted_user] = [item + item_offset for item in filtered_valid_user_list[user]]
                    
                if user in filtered_test_user_list:
                    self.test_user_list[adjusted_user] = [item + item_offset for item in filtered_test_user_list[user]]
                
                # Add test negatives with adjusted offsets
                if user in filtered_test_neg_user_list:
                    self.test_neg_user_list[adjusted_user] = [item + item_offset for item in filtered_test_neg_user_list[user]]
            
            # Add items with offset
            for item in dataset_items:
                all_items.add(item + item_offset)
            
            # Update offsets for next dataset
            user_offset += len(dataset_users)
            item_offset += len(dataset_items)
        
        # Set combined dataset properties
        # Users are already consecutive integers due to offset mechanism
        self.users = list(range(len(all_users)))
        self.items = sorted(list(all_items))
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        self.n_observations = sum(len(items) for items in self.train_user_list.values())
        
        print(f"Combined dataset: {self.n_users} users, {self.n_items} items, {self.n_observations} interactions")
        
        # Initialize popularity and other metrics
        self._compute_popularity_metrics()
        
        # Calculate dataset sampling weights
        if self.dataset_sampling_weights:
            # Custom weights override everything
            if len(self.dataset_sampling_weights) != len(self.multi_datasets):
                raise ValueError("Number of dataset sampling weights must match number of datasets")
            self.dataset_weights = dict(zip(self.multi_datasets, self.dataset_sampling_weights))
        elif self.proportional_sampling:
            # Proportional to dataset sizes
            total_interactions = sum(info['n_interactions'] for info in self.dataset_info.values())
            self.dataset_weights = {
                name: info['n_interactions'] / total_interactions 
                for name, info in self.dataset_info.items()
            }
        elif self.equal_sampling:
            # Equal weights
            self.dataset_weights = {name: 1.0 / len(self.multi_datasets) for name in self.multi_datasets}
        else:
            # Default to proportional sampling
            total_interactions = sum(info['n_interactions'] for info in self.dataset_info.values())
            self.dataset_weights = {
                name: info['n_interactions'] / total_interactions 
                for name, info in self.dataset_info.items()
            }
            
        print(f"Dataset sampling weights: {self.dataset_weights}")
        
        # Set sample_items for negative sampling
        self.sample_items = np.array(self.items, dtype=int)
        
        # Clear unused attributes
        self.selected_train, self.selected_valid, self.selected_test = [], [], []
        self.nu_info = []
        self.ni_info = []
        
    def _compute_popularity_metrics(self):
        """Compute popularity metrics for the combined dataset"""
        # Build item popularity
        train_item_list = collections.defaultdict(list)
        for user, items in self.train_user_list.items():
            for item in items:
                train_item_list[item].append(user)
        
        self.train_item_list = train_item_list
        
        # Compute popularity
        pop_dict = {}
        for item, users in train_item_list.items():
            pop_dict[item] = len(users) + 1
        for item in range(0, self.n_items):
            if item not in pop_dict:
                pop_dict[item] = 1
        
        self.population_list = [pop_dict[item] for item in range(self.n_items)]
        
        # User and item popularity indices
        pop_user = {key: len(value) for key, value in self.train_user_list.items()}
        pop_item = {key: len(value) for key, value in train_item_list.items()}
        
        # Convert to indices
        sorted_pop_user = sorted(list(set(pop_user.values())))
        sorted_pop_item = sorted(list(set(pop_item.values())))
        
        user_idx = {val: idx for idx, val in enumerate(sorted_pop_user)}
        item_idx = {val: idx for idx, val in enumerate(sorted_pop_item)}
        
        self.user_pop_idx = np.zeros(self.n_users, dtype=int)
        self.item_pop_idx = np.zeros(self.n_items, dtype=int)
        
        for user, pop in pop_user.items():
            self.user_pop_idx[user] = user_idx[pop]
        for item, pop in pop_item.items():
            self.item_pop_idx[item] = item_idx[pop]
            
        self.user_pop_max = max(self.user_pop_idx)
        self.item_pop_max = max(self.item_pop_idx)
        
    def get_user_dataset(self, user_id):
        """Get which dataset a user belongs to"""
        for dataset_name, info in self.dataset_info.items():
            if info['user_offset'] <= user_id < info['user_offset'] + info['n_users']:
                return dataset_name
        return None
        
    def get_item_dataset(self, item_id):
        """Get which dataset an item belongs to"""
        for dataset_name, info in self.dataset_info.items():
            if info['item_offset'] <= item_id < info['item_offset'] + info['n_items']:
                return dataset_name
        return None
        
    def get_dataset_items(self, dataset_name):
        """Get all item IDs for a specific dataset"""
        info = self.dataset_info[dataset_name]
        return list(range(info['item_offset'], info['item_offset'] + info['n_items']))
        
    def get_dataloader(self):
        """Create multi-dataset aware dataloader"""
        self.train_data = MultiDatasetTrainDataset(
            self.model_name, self.users, self.train_user_list, self.user_pop_idx,
            self.item_pop_idx, self.neg_sample, self.n_observations, self.n_items, 
            self.sample_items, self.infonce, self.items, self.nu_info, self.ni_info, 
            self.is_one_pos_item, n_pos_samples=self.n_pos_samples, 
            user_neighbors=self.user_neighbors, dataset_info=self.dataset_info
        )

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, drop_last=True)


class MultiDatasetTrainDataset(TrainDataset):
    """
    Multi-dataset training dataset that ensures negative sampling is done within the same dataset
    """
    
    def __init__(self, model_name, users, train_user_list, user_pop_idx, item_pop_idx, neg_sample,
                 n_observations, n_items, sample_items, infonce, items, nu_info=None, ni_info=None,
                 is_one_pos_item=True, n_pos_samples=15, user_neighbors=None, dataset_info=None):
        super().__init__(model_name, users, train_user_list, user_pop_idx, item_pop_idx, neg_sample,
                         n_observations, n_items, sample_items, infonce, items, nu_info, ni_info,
                         is_one_pos_item, n_pos_samples, user_neighbors)
        
        self.dataset_info = dataset_info or {}
        
        # Create mapping from user to dataset
        self.user_to_dataset = {}
        for dataset_name, info in self.dataset_info.items():
            for user_id in range(info['user_offset'], info['user_offset'] + info['n_users']):
                self.user_to_dataset[user_id] = dataset_name
    
    def __getitem__(self, index):
        index = index % len(self.users)
        user = self.users[index]
        
        if self.train_user_list[user] == []:
            pos_items = 0
            mask = 0
        else:
            if self.is_one_pos_item:
                pos_item = rd.choice(self.train_user_list[user])
                mask = 1
            else:
                pos_item = rd.sample(self.train_user_list[user], self.n_pos_samples) if \
                    len(self.train_user_list[user]) > self.n_pos_samples else \
                    self.train_user_list[user][:self.n_pos_samples]
                mask = torch.zeros(self.n_pos_samples).long()
                mask[:len(pos_item)] = 1
                if len(pos_item) < self.n_pos_samples:
                    pos_item += [-1] * (self.n_pos_samples - len(pos_item))
                pos_item = torch.tensor(pos_item).long()
        
        user_pop = self.user_pop_idx[user]
        if self.is_one_pos_item:
            pos_item_pop = self.item_pop_idx[pos_item]
        else:
            pos_item_pop = -1
        
        if self.infonce == 1 and self.neg_sample == -1:  # in-batch
            return user, pos_item, user_pop, pos_item_pop

        elif self.infonce == 1 and self.neg_sample != -1:  # InfoNCE negative sampling
            # Multi-dataset aware negative sampling
            user_dataset = self.user_to_dataset.get(user)
            if user_dataset and user_dataset in self.dataset_info:
                # Get items only from the same dataset
                dataset_info = self.dataset_info[user_dataset]
                dataset_items = list(range(dataset_info['item_offset'], 
                                         dataset_info['item_offset'] + dataset_info['n_items']))
                
                # Sample negative items only from the same dataset
                neg_items = randint_choice(len(dataset_items), size=self.neg_sample, 
                                         exclusion=[item - dataset_info['item_offset'] 
                                                   for item in self.train_user_list[user]])
                neg_items = [dataset_items[idx] for idx in neg_items]
            else:
                # Fallback to original behavior
                neg_items = randint_choice(self.n_items, size=self.neg_sample, 
                                         exclusion=self.train_user_list[user])
            
            neg_items_pop = self.item_pop_idx[neg_items]
            return user, pos_item, user_pop, pos_item_pop, torch.tensor(neg_items).long(), neg_items_pop, mask

        else:  # BPR negative sampling
            # Multi-dataset aware negative sampling for BPR
            user_dataset = self.user_to_dataset.get(user)
            if user_dataset and user_dataset in self.dataset_info:
                dataset_info = self.dataset_info[user_dataset]
                dataset_items = list(range(dataset_info['item_offset'], 
                                         dataset_info['item_offset'] + dataset_info['n_items']))
                
                # Sample one negative item from the same dataset
                while True:
                    idx = rd.randint(0, len(dataset_items) - 1)
                    neg_item = dataset_items[idx]
                    if neg_item not in self.train_user_list[user]:
                        break
            else:
                # Fallback to original behavior
                while True:
                    idx = rd.randint(0, self.n_items - 1)
                    neg_item = self.items[idx]
                    if neg_item not in self.train_user_list[user]:
                        break
            
            neg_item_pop = self.item_pop_idx[neg_item]
            return user, pos_item, user_pop, pos_item_pop, neg_item, neg_item_pop
