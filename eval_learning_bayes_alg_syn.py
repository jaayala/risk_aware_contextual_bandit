import os
import sys
import json
from datetime import datetime
from argparse import ArgumentParser
sys.path.append(os.path.join(os.getcwd(), 'learning_algorithms'))
import numpy as np

from bayesian_OL import Bayesian_OL
from env_syn import EnvSyn


parser = ArgumentParser()
parser.add_argument("--algo", nargs='+', default=['safeOPT'])
parser.add_argument("--train_iters", type=int, default=1000)
parser.add_argument("--eval_iters", type=int, default=250)
parser.add_argument("--beta", type=float, default=10)
parser.add_argument("-T", "--timestamp", nargs='+', default=[''])
parser.add_argument("-t", "--tag", nargs='+', default=[''])
parser.add_argument("--env_noise_var", type=float, default=0.2)
parser.add_argument("-v", "--version", type=int, default=2)
parser.add_argument("--constr_val", nargs='+', default=['[0.3, 0.3]'])

args = parser.parse_args()
algo = args.algo[0]
beta = args.beta
timestamp = args.timestamp[0]
tag = args.tag[0]
train_iters = args.train_iters
eval_iters = args.eval_iters
env_noise_var = args.env_noise_var
env_version = args.version

dataset_n_points, dataset_actions_per_context = 1000, 10

constr_val = json.loads(' '.join(args.constr_val))
tag = '_'+tag if len(tag)>0 else tag

timestamp = datetime.now().strftime("%Y%m%d%H%M%S") if timestamp=='' else timestamp

if env_version == 1:
    contexts_ts_training = '20230306160703'
    contexts_ts_eval = '20230306160929'
elif env_version == 2:
    contexts_ts_training = '20230626165918'
    contexts_ts_eval = '20230626165924'

env_noise = {'dist' : 'norm', 'var' : env_noise_var}

contexts_file = os.path.join(os.getcwd(), 'datasets', contexts_ts_training, contexts_ts_training+'_context_log.csv')
env = EnvSyn(algo=algo+'_train'+tag, noise=env_noise, timestamp=timestamp, contexts_file=contexts_file, ver=env_version)

context_actions, rew, cons = env.gen_dataset(dataset_n_points, dataset_actions_per_context)

######## Save config data ########
config_file = os.path.join(os.getcwd(), 'results', timestamp, timestamp+f'_{algo+tag}_config.json')

config_data = {'algorithm' : algo, 'beta' : beta, 'train_iters' : train_iters, 
               'eval_iters' : eval_iters, 'timestamp' : timestamp, 'env_noise' : str(env_noise), 
              'constr_val' : constr_val, 'contexts_ts_training' : contexts_ts_training, 
               'contexts_ts_eval' : contexts_ts_eval,  }
with open(config_file, 'w') as f:
    json.dump(config_data, f, indent=4)
################################

if hasattr(constr_val, "__len__"):
    constr_greater = np.zeros(len(constr_val))
else:
    constr_greater = 0

contexts = {'c1': '', 'c2': '', 'c3': ''} if env_version == 2 else {'c1': '', 'c2': ''}
bol = Bayesian_OL(env, contexts=contexts, beta_val=beta, constr_val=constr_val, constr_greater=constr_greater, points=context_actions, rewards=rew, constraint_vals=cons)


bol.run(train_iters)
bol.env.close()

contexts_file_eval = os.path.join(os.getcwd(), 'datasets', contexts_ts_eval, contexts_ts_eval+'_context_log.csv')
env_eval = EnvSyn(algo=algo+tag, noise=env_noise, timestamp=timestamp, contexts_file=contexts_file_eval, ver=env_version)
bol.env = env_eval
bol.run(eval_iters, evaluation=1)

bol.close()






