import numpy as np
import torch
import torch.nn as nn
# from data import Data
from .evaluator import ProxyEvaluator
from .util import DataIterator
from .util.cython.tools import float_type, is_ndarray
from .util import typeassert, argmax_top_k
from concurrent.futures import ThreadPoolExecutor
from .utils import *
import datetime
import json
import wandb
from .abstract_data import AbstractData

# define the abstract class for recommender system
class AbstractRS(nn.Module):
    def __init__(self, args, special_args) -> None:
        super(AbstractRS, self).__init__()
        # print(torch.cuda.device_count())
        # # 查看当前设备索引号
        # print(torch.cuda.current_device())
        # # 根据索引号查看设备名
        # print(torch.cuda.get_device_name(0))
        # basic information
        self.args = args
        self.special_args = special_args
        self.device = torch.device(f"cuda:{args.cuda}" if args.cuda != -1 and torch.cuda.is_available() else "cpu")
        self.test_only = args.test_only
        self.candidate = args.candidate

        self.Ks = args.Ks
        self.patience = args.patience
        self.model_name = args.model_name
        self.neg_sample = args.neg_sample
        self.inbatch = self.args.infonce == 1 and self.args.neg_sample == -1
    
        # basic hyperparameters
        self.n_layers = args.n_layers
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.verbose = args.verbose

        self.mix = True if 'mix' in args.dataset else False

        # load the data
        self.dataset_name = args.dataset
        # from models.General.IntentCF import IntentCF_Data
        # self.data = IntentCF_Data(args)
        # from models.General.UniSRec import UniSRec_Data
        # self.data = UniSRec_Data(args)
        try:
            print('from models.General.'+ args.model_name + ' import ' + args.model_name + '_Data')
            exec('from models.General.'+ args.model_name + ' import ' + args.model_name + '_Data') # load special dataset
            self.data = eval(args.model_name + '_Data(args)') 
        except:
            print("no special dataset")
            # Check if we're using multi-dataset training
            if hasattr(args, 'multi_datasets') and args.multi_datasets:
                print("Using multi-dataset training")
                from .abstract_data import MultiDatasetData
                self.data = MultiDatasetData(args)
            else:
                self.data = AbstractData(args) # load data from the path
        
        self.n_users = self.data.n_users
        self.n_items = self.data.n_items
        self.train_user_list = self.data.train_user_list
        self.valid_user_list = self.data.valid_user_list
        # = torch.tensor(self.data.population_list).to(self.device)
        self.user_pop = torch.tensor(self.data.user_pop_idx).type(torch.LongTensor).to(self.device)
        self.item_pop = torch.tensor(self.data.item_pop_idx).type(torch.LongTensor).to(self.device)
        self.user_pop_max = self.data.user_pop_max
        self.item_pop_max = self.data.item_pop_max 

        # load the model
        self.running_model = args.model_name + '_batch' if self.inbatch else args.model_name
        # from models.General.IntentCF import IntentCF
        # self.model = IntentCF(args, self.data) # initialize the model with the graph
        exec('from models.General.'+ args.model_name + ' import ' + self.running_model) # import the model first
        self.model = eval(self.running_model + '(args, self.data)') # initialize the model with the graph
        self.model.to(self.device)

        # preparing for saving
        self.preperation_for_saving(args, special_args)
        
        # preparing for evaluation
        # self.not_candidate_dict = self.data.get_not_candidate() # load the not candidate dict
        self.evaluators, self.eval_names = self.get_evaluators(self.data) # load the evaluators


    # the whole pipeline of the training process
    def execute(self):
        
        self.save_args() # save the args
        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.device) # restore the checkpoint

        start_time = time.time()
        # train the model if not test only
        if not self.test_only:
            print("start training") 
            self.train()
            # test the model
            print("start testing")
            self.model = self.restore_best_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.device)
        end_time = time.time()
        print(f'training time: {end_time - start_time}')
        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d, total training cost is %.1f" % (max(self.data.best_valid_epoch, self.start_epoch), end_time - start_time)
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        n_rets = {}
        start_time = time.time()
        for i,evaluator in enumerate(self.evaluators[:]):
            _, __, n_ret = evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, evaluator, self.eval_names[i])
            n_rets[self.eval_names[i]] = n_ret
        end_time = time.time()
        print(f'evaluation time: {end_time - start_time}')
        self.recommend_top_k()
        # self.document_hyper_params_results(self.base_path, n_rets)

    def save_args(self):
        # save the args
        with open(self.base_path + '/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

    # define the training process
    def train(self) -> None:
        # TODO
        self.set_optimizer() # get the optimizer
        self.flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            # print(self.model.embed_user.weight)
            if self.flag: # early stop
                break
            # All models
            t1=time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2=time.time()
            print(f'epoch={epoch};  loss={losses}')
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch)

        visualize_and_save_log(self.base_path +'stats.txt', self.dataset_name)

    #! must be implemented by the subclass
    def train_one_epoch(self, epoch):
        raise NotImplementedError
    
    def preperation_for_saving(self, args, special_args):
        self.formatted_today=datetime.date.today().strftime('%m%d') + '_'

        tn = '1' if args.train_norm else '0'
        pn = '1' if args.pred_norm else '0'
        self.train_pred_mode = 't' + tn + 'p' + pn

        if(self.test_only == False):
            prefix = self.formatted_today + args.saveID
        else:
            prefix = args.saveID
        self.saveID = prefix + '_' + self.train_pred_mode + "_Ks=" + str(args.Ks) + '_patience=' + str(args.patience)\
            + "_n_layers=" + str(args.n_layers) + "_batch_size=" + str(args.batch_size)\
                + "_neg_sample=" + str(args.neg_sample) + "_lr=" + str(args.lr) + "_hidden_size=" + str(args.hidden_size)
        
        for arg in special_args:
            print(arg, getattr(args, arg))
            self.saveID += "_" + arg + "=" + str(getattr(args, arg))
        
        self.modify_saveID()

        if self.model_name == 'LightGCN' and self.n_layers == 0:
            self.base_path = './weights/General/{}/MF/{}'.format(self.dataset_name, self.saveID)
        elif self.n_layers > 0 and self.model_name != "LightGCN":
            self.base_path = './weights/General/{}/{}-LGN/{}'.format(self.dataset_name, self.running_model, self.saveID)
        else:
            self.base_path = './weights/General/{}/{}/{}'.format(self.dataset_name, self.running_model, self.saveID)
        self.checkpoint_buffer=[]
        ensureDir(self.base_path)

    def modify_saveID(self):
        pass

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], lr=self.lr)

    def document_running_loss(self, losses:list, epoch, t_one_epoch, prefix=""):
        loss_str = ', '.join(['%.5f']*len(losses)) % tuple(losses)
        perf_str = prefix + 'Epoch %d [%.1fs]: train==[' % (
                epoch, t_one_epoch) + loss_str + ']'
        with open(self.base_path + 'stats.txt','a') as f:
                f.write(perf_str+"\n")
    
    def document_hyper_params_results(self, base_path, n_rets):
        overall_path = '/'.join(base_path.split('/')[:-1]) + '/'
        hyper_params_results_path = overall_path + self.formatted_today + self.dataset_name + '_' + self.model_name + '_' + self.args.saveID + '_hyper_params_results.csv'

        results = {'notation': self.formatted_today, 'train_pred_mode':self.train_pred_mode, 'best_epoch': max(self.data.best_valid_epoch, self.start_epoch), 'max_epoch': self.max_epoch, 'Ks': self.Ks, 'n_layers': self.n_layers, 'batch_size': self.batch_size, 'neg_sample': self.neg_sample, 'lr': self.lr}
        for special_arg in self.special_args:
            results[special_arg] = getattr(self.args, special_arg)

        for k, v in n_rets.items():
            if('test_id' not in k):
                for metric in ['recall', 'ndcg', 'hit_ratio']:
                    results[k + '_' + metric] = round(v[metric], 4)
        frame_columns = list(results.keys())
        # load former xlsx
        if os.path.exists(hyper_params_results_path):
            # hyper_params_results = pd.read_excel(hyper_params_results_path)
            hyper_params_results = pd.read_csv(hyper_params_results_path)
        else:
            # Create a new dataframe using the results.
            hyper_params_results = pd.DataFrame(columns=frame_columns)

        hyper_params_results = hyper_params_results._append(results, ignore_index=True)
        # to csv
        hyper_params_results.to_csv(hyper_params_results_path, index=False, float_format='%.4f')
        # hyper_params_results.to_excel(hyper_params_results_path, index=False)

    def recommend_top_k(self):
        test_users = list(self.data.test_user_list.keys())
        if self.args.nodrop: # whether using the enhanced dataset
            eval_train_user_list = self.data.train_nodrop_user_list
        else:
            eval_train_user_list = self.data.train_user_list
        if(self.candidate == False):
            dump_dict = merge_user_list([eval_train_user_list,self.data.valid_user_list])
        recommended_top_k = {}
        recommended_scores = {}
        test_users = DataIterator(test_users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        for batch_id, batch_users in enumerate(test_users):
            if self.data.test_neg_user_list is not None:
                candidate_items = {u:list(self.data.test_user_list[u]) + self.data.test_neg_user_list[u] if u in self.data.test_neg_user_list.keys() else list(self.data.test_user_list[u]) for u in batch_users}

                ranking_score = self.model.predict(batch_users, None)  # (B,N)
                if not is_ndarray(ranking_score, float_type):
                    ranking_score = np.array(ranking_score, dtype=float_type)

                all_items = set(range(ranking_score.shape[1]))
                for idx, user in enumerate(batch_users):
                    # print(max(set(candidate_items[user])), )
                    not_user_candidates = list(all_items - set(candidate_items[user]))
                    ranking_score[idx,not_user_candidates] = -np.inf

                    pos_items = self.data.valid_user_list[user]
                    pos_items = [ x for x in pos_items if not x in self.data.test_user_list[user] ]
                    ranking_score[idx][pos_items] = -np.inf

                    recommended_top_k[user] = argmax_top_k(ranking_score[idx], self.Ks)
                    # ground_truth = self.data.test_user_list[user]
                    # hits = [1 if item in ground_truth else 0 for item in recommended_top_k[user]]
                    # print(sum(hits)/self.Ks)
                    recommended_scores[user] = ranking_score[idx][recommended_top_k[user]]
                    # print('finish one user')
            else:
                ranking_score = self.model.predict(batch_users, None)  # (B,N)
                if not is_ndarray(ranking_score, float_type):
                    ranking_score = np.array(ranking_score, dtype=float_type)
                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.
                
                for idx, user in enumerate(batch_users):
                    dump_items = dump_dict[user]
                    dump_items = [ x for x in dump_items if not x in self.data.test_user_list[user] ]
                    ranking_score[idx][dump_items] = -np.inf

                    recommended_top_k[user] = argmax_top_k(ranking_score[idx], self.Ks)
                    recommended_scores[user] = ranking_score[idx][recommended_top_k[user]]
                    # recommended_scores[user] = ranking_score[idx]
            print('finish recommend one batch', batch_id)

        # 保存rank score
        with open(self.base_path + '/recommend_top_k.txt', 'w') as f:
            for u, v in recommended_top_k.items():
                f.write(str(int(u)))
                for i in range(self.Ks):
                    f.write(' ' + str(int(v[i])))
                f.write('\n')
        with open(self.base_path + '/recommend_top_k_with_score.txt', 'w') as f:
            for u, v in recommended_top_k.items():
                f.write(str(int(u)))
                for i in range(self.Ks):
                    f.write(' ' + str(int(v[i])) + '+' + str(round(recommended_scores[u][i], 4)))
                f.write('\n')
        print('finish recommend top k')
    
    # define the evaluation process
    def eval_and_check_early_stop(self, epoch):
        self.model.eval()

        # During training, only evaluate on the combined dataset (first two evaluators)
        if not self.test_only:
            # Only use the first two evaluators (valid and test for combined dataset)
            for i in range(min(2, len(self.evaluators))):
                tt1 = time.time()
                is_best, temp_flag, n_ret = evaluation(self.args, self.data, self.model, epoch, self.base_path, self.evaluators[i], self.eval_names[i])
                tt2 = time.time()
                print("Evaluating %d [%.1fs]: %s" % (i, tt2 - tt1, self.eval_names[i]))
                if(not self.args.no_wandb):
                    wandb.log(
                        data = {f"Recall@{self.Ks}": n_ret['recall'], 
                                f"Hit Ratio@{self.Ks}": n_ret['recall'],
                                f"Precision@{self.Ks}": n_ret['precision'],
                                f"NDCG@{self.Ks}": n_ret['ndcg']},
                        step = epoch
                    )
                if is_best:
                    checkpoint_buffer=save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)
                
                # early stop?
                if temp_flag:
                    self.flag = True
        else:
            # During testing, evaluate on all datasets
            for i, evaluator in enumerate(self.evaluators):
                tt1 = time.time()
                is_best, temp_flag, n_ret = evaluation(self.args, self.data, self.model, epoch, self.base_path, evaluator, self.eval_names[i])
                tt2 = time.time()
                print("Evaluating %d [%.1fs]: %s" % (i, tt2 - tt1, self.eval_names[i]))
                if(not self.args.no_wandb):
                    wandb.log(
                        data = {f"Recall@{self.Ks}": n_ret['recall'], 
                                f"Hit Ratio@{self.Ks}": n_ret['recall'],
                                f"Precision@{self.Ks}": n_ret['precision'],
                                f"NDCG@{self.Ks}": n_ret['ndcg']},
                        step = epoch
                    )
                if is_best:
                    checkpoint_buffer=save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)
                
                # early stop?
                if temp_flag:
                    self.flag = True
        
        self.model.train()
    
    # load the checkpoint
    def restore_checkpoint(self, model, checkpoint_dir, device, force=False, pretrain=False):
        """
        If a checkpoint exists, restores the PyTorch model from the checkpoint.
        Returns the model and the current epoch.
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        if not cp_files:
            print('No saved model parameters found')
            if force:
                raise Exception("Checkpoint not found")
            else:
                return model, 0,

        epoch_list = []

        regex = re.compile(r'\d+')

        for cp in cp_files:
            epoch_list.append([int(x) for x in regex.findall(cp)][0])

        epoch = max(epoch_list)

        if not force:
            print("Which epoch to load from? Choose in range [0, {})."
                .format(epoch), "Enter 0 to train from scratch.")
            print(">> ", end = '')
            # inp_epoch = int(input())

            if self.args.clear_checkpoints:
                print("Clear checkpoint")
                clear_checkpoint(checkpoint_dir)
                return model, 0,

            inp_epoch = epoch
            if inp_epoch not in range(epoch + 1):
                raise Exception("Invalid epoch number")
            if inp_epoch == 0:
                print("Checkpoint not loaded")
                clear_checkpoint(checkpoint_dir)
                return model, 0,
        else:
            print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
            inp_epoch = int(input())
            if inp_epoch not in range(0, epoch):
                raise Exception("Invalid epoch number")

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))
        # print("finish load")

        try:
            if pretrain:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)"
                .format(checkpoint['epoch']))
        except:
            print("=> Checkpoint not successfully restored")
            raise

        return model, inp_epoch
    
    def restore_best_checkpoint(self, epoch, model, checkpoint_dir, device):
        """
        Restore the best performance checkpoint
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))

        model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))

        return model
    



    def get_evaluators(self, data):
        #if not self.args.pop_test:
        K_value = self.args.Ks
        if self.args.nodrop: # whether using the enhanced dataset
            eval_train_user_list = data.train_nodrop_user_list
        else:
            eval_train_user_list = data.train_user_list

        if self.mix:
            eval_valid = ProxyEvaluator(data,eval_train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.test_user_list]))  
            eval_test = ProxyEvaluator(data,eval_train_user_list,data.test_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.valid_user_list]))

            evaluators=[eval_valid, eval_test]
            eval_names=["valid", "test"]
            
            for i, data_name in enumerate(self.data.mixed_datasets):
                mask_ = list(set(list(range(self.data.n_items))) - set(list(range(self.data.cum_ni_info[i], self.data.cum_ni_info[i+1]))))
                eval_valid_ = ProxyEvaluator(data,data.selected_train[i],data.selected_valid[i],top_k=[K_value],dump_dict=merge_user_list([data.selected_train[i], data.selected_test[i]]), masked_items=mask_)
                eval_test_ = ProxyEvaluator(data,data.selected_train[i],data.selected_test[i],top_k=[K_value],dump_dict=merge_user_list([data.selected_train[i], data.selected_valid[i]]), masked_items=mask_)
                evaluators.append(eval_valid_)
                evaluators.append(eval_test_)
                eval_names.append(data_name + "_valid")
                eval_names.append(data_name + "_test")
                
        elif hasattr(data, 'dataset_info') and data.dataset_info:  # Multi-dataset training
            # Combined evaluation
            eval_valid = ProxyEvaluator(data,eval_train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.test_user_list]))  
            eval_test = ProxyEvaluator(data,eval_train_user_list,data.test_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.valid_user_list]))

            evaluators=[eval_valid, eval_test]
            eval_names=["valid", "test"]
            
            # Per-dataset evaluation
            for dataset_name, dataset_info in data.dataset_info.items():
                # Create masked items (exclude items from other datasets)
                dataset_item_range = set(range(dataset_info['item_offset'], 
                                             dataset_info['item_offset'] + dataset_info['n_items']))
                mask_ = list(set(range(data.n_items)) - dataset_item_range)
                
                # Create user mappings for this dataset
                dataset_train = {}
                dataset_valid = {}
                dataset_test = {}
                
                for user_id in range(dataset_info['user_offset'], 
                                   dataset_info['user_offset'] + dataset_info['n_users']):
                    if user_id in data.train_user_list:
                        dataset_train[user_id] = data.train_user_list[user_id]
                    if user_id in data.valid_user_list:
                        dataset_valid[user_id] = data.valid_user_list[user_id]
                    if user_id in data.test_user_list:
                        dataset_test[user_id] = data.test_user_list[user_id]
                
                # Create evaluators for this dataset
                if dataset_valid:
                    eval_valid_ = ProxyEvaluator(data, dataset_train, dataset_valid, top_k=[K_value],
                                               dump_dict=merge_user_list([dataset_train, dataset_test]),
                                               masked_items=mask_)
                    evaluators.append(eval_valid_)
                    eval_names.append(f"{dataset_name}_valid")
                
                if dataset_test:
                    eval_test_ = ProxyEvaluator(data, dataset_train, dataset_test, top_k=[K_value],
                                              dump_dict=merge_user_list([dataset_train, dataset_valid]),
                                              masked_items=mask_)
                    evaluators.append(eval_test_)
                    eval_names.append(f"{dataset_name}_test")
        else: 
            eval_valid = ProxyEvaluator(data,eval_train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.test_user_list]))  
            eval_test = ProxyEvaluator(data,eval_train_user_list,data.test_user_list,top_k=[K_value],dump_dict=merge_user_list([eval_train_user_list, data.valid_user_list]))

            evaluators=[eval_valid, eval_test]
            eval_names=["valid", "test"]

        return evaluators, eval_names

