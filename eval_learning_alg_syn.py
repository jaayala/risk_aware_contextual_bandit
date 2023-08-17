import os
import sys
import json
from datetime import datetime
from argparse import ArgumentParser
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.getcwd(), 'learning_algorithms'))

from distributional_cb import DistributionalCB
from cost_functions import constrained_loss, critic_objective
from env_syn import EnvSyn
from utils_learning import OUNoise



parser = ArgumentParser()
parser.add_argument("--algo", nargs='+', default=['RANCB'])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--train_iters", type=int, default=3000)
parser.add_argument("--eval_iters", type=int, default=500)
parser.add_argument("--updates_per_step", type=int, default=5)
parser.add_argument("-T", "--timestamp", nargs='+', default=[''])
parser.add_argument("-t", "--tag", nargs='+', default=[''])
parser.add_argument("--env_noise_var", type=float, default=0.1)
parser.add_argument("--constr_val", nargs='+', default=['[0.3, 0.3]'])
parser.add_argument("-L", "--landa", type=float, default=2.5)
parser.add_argument("-v", "--version", type=int, default=2)

default_alphas = '{"A_exec" : [0.5], "A_train" : [0.5], "A_explo" : [0.5]}'
parser.add_argument("-a", "--alpha", nargs='+', default=[default_alphas])
parser.add_argument("--quant_critic", nargs='+', default=['{"quantile_critic" : [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99, 0.995, 0.999]}'])

args = parser.parse_args()
algo = args.algo[0]
updates_per_step = args.updates_per_step
timestamp = args.timestamp[0]
tag = args.tag[0]
batch_size = args.batch_size
train_iters = args.train_iters
eval_iters = args.eval_iters
env_noise_var = args.env_noise_var
env_version = args.version

landa = args.landa
quant_critic = ' '.join(args.quant_critic)

alpha_dict = ' '.join(args.alpha)
alpha_dict = json.loads(alpha_dict)
constr_val = json.loads(' '.join(args.constr_val))

assert 'A_exec' in alpha_dict and 'A_train' in alpha_dict and 'A_explo' in alpha_dict
A_exec = alpha_dict['A_exec']
A_train = alpha_dict['A_train']
A_explo = alpha_dict['A_explo']


alpha_all = []
for k,v in alpha_dict.items():
    alpha_all+=v
    
if len(quant_critic) == 0:
    quantile_critic = sorted(list(set(alpha_all)))
else:
    quantile_critic_dict = json.loads(quant_critic)
    quantile_critic = quantile_critic_dict['quantile_critic']
    
    for a in alpha_all:  #sanity check
        assert a in quantile_critic
    

tag = '_'+tag if len(tag)>0 else tag

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'torch device: {device}')

timestamp = datetime.now().strftime("%Y%m%d%H%M%S") if timestamp=='' else timestamp


if env_version == 1:
    contexts_ts_training = '20230306160703'
    contexts_ts_eval = '20230306160929'
elif env_version == 2:
    contexts_ts_training = '20230626165918'
    contexts_ts_eval = '20230626165924'

cnn_kwargs = None
# cnn_kwargs = {'in_features' : 1, 'num_cells' : [32, 8, 2], 'kernel_sizes' : 5}
# cnn_kwargs = {'in_features' : 1, 'num_cells' : [1], 'kernel_sizes' : 7, 'norm_class' : nn.MaxPool3d, 'norm_kwargs' : {'kernel_size' : 2, 'stride' : 2}}
NN_arch = [256, 256]
distrib_critic = 0

if algo == "RANCB":
    critic_kwargs_1 = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU, 'n_quantiles' : 21, 'quantile_vals' : None}
    critic_kwargs_2 = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU, 'n_quantiles' : len(quantile_critic), 'quantile_vals' : quantile_critic}
    critic_kwargs_3 = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU, 'n_quantiles' : len(quantile_critic), 'quantile_vals' : quantile_critic}
    critics_kwargs = [critic_kwargs_1, critic_kwargs_2]
    if env_version == 2:
        critics_kwargs.append(critic_kwargs_3)
    critic_loss = 'quantile_huber_loss'
    distrib_critic = 1
    
elif algo == "NCB":
    critic_kwargs = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU}
    critics_kwargs = [critic_kwargs]
    critic_loss = 'MSELoss'

    
elif algo == "SC_DNCB":
    critic_kwargs = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU, 'n_quantiles' : 21, 'quantile_vals' : None}
    critics_kwargs = [critic_kwargs]
    critic_loss = 'quantile_huber_loss'
    distrib_critic = 1

    
elif algo == "MC_NCB":
    critic_kwargs_1 = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU}
    critic_kwargs_2 = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU}
    critic_kwargs_3 = {'num_cells' : NN_arch, 'activation_class' : nn.ReLU}
    critics_kwargs = [critic_kwargs_1, critic_kwargs_2]
    if env_version == 2:
        critics_kwargs.append(critic_kwargs_3)
    critic_loss = 'MSELoss'
else:
    raise Exception('Unknown algorithm')

actor_kwargs = {'num_cells' : [256, 256], 'activation_class' : nn.ReLU, 'last_activation' : nn.Sigmoid}

noise = OUNoise(1, decay_period=train_iters)
env_noise = {'dist' : 'norm', 'var' : env_noise_var}


contexts_file = os.path.join(os.getcwd(), 'datasets', contexts_ts_training, contexts_ts_training+'_context_log.csv')
env = EnvSyn(algo=algo+'_train'+tag, noise=env_noise, timestamp=timestamp, contexts_file=contexts_file, ver=env_version)


actor_loss = constrained_loss(constr_val, landa, distrib_critic, quantile_critic, constr_op='max', device=device)
critic_objective_class = critic_objective(n_critics=len(critics_kwargs), landa=landa, constr_val=constr_val, constr_op='max', device=device)
alpha_input_actor = 1 if algo == "DNCB" else 0


######## Save config data ########
config_file = os.path.join(os.getcwd(), 'results', timestamp, timestamp+f'_{algo+tag}_config.json')

config_data = {'algorithm' : algo, 'updates_per_step' : updates_per_step, 'train_iters' : train_iters, 
               'eval_iters' : eval_iters, 'timestamp' : timestamp, 'env_noise' : str(env_noise), 
               'landa' : landa, 'quantile_critic' : quantile_critic, 'A_exec' : A_exec, 'A_train' : A_train,
               'A_explo' : A_explo, 'constr_val' : constr_val, 'contexts_ts_training' : contexts_ts_training, 
               'contexts_ts_eval' : contexts_ts_eval, 'critic_loss' : critic_loss, 
               'critic_objective_class' : str(critic_objective_class), 'actor_loss' : str(actor_loss), 
               'alpha_input_actor' : alpha_input_actor, 'device' : str(device), 
               'cnn_kwargs' : str(cnn_kwargs), 'actor_kwargs' : str(actor_kwargs), 'critics_kwargs' : str(critics_kwargs), 
               'exploration_noise' : str(type(noise)) }
with open(config_file, 'w') as f:
    json.dump(config_data, f, indent=4)
################################

distrib_cb = DistributionalCB(env, noise, batch_size, actor_kwargs=actor_kwargs, critics_kwargs=critics_kwargs, actor_loss_function=actor_loss, 
                              critic_loss=critic_loss, critic_objective_class=critic_objective_class,  A_train=A_train, A_explo=A_explo, landa=landa,
                              alpha_input_actor=alpha_input_actor, convnet_kwargs=cnn_kwargs, updates_per_step=updates_per_step, device=device)


distrib_cb.run(train_iters)
distrib_cb.env.close()

contexts_file_eval = os.path.join(os.getcwd(), 'datasets', contexts_ts_eval, contexts_ts_eval+'_context_log.csv')
env_eval = EnvSyn(algo=algo+tag, noise=env_noise, timestamp=timestamp, contexts_file=contexts_file_eval, ver=env_version)
distrib_cb.env = env_eval

for alpha_exec in A_exec:
    distrib_cb.run(eval_iters, evaluation=1, alpha_exec=alpha_exec)
distrib_cb.close()





