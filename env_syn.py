import os
from datetime import datetime
import csv
from gym import spaces
import numpy as np


class csv_handler:
    def __init__(self, filename, fieldnames):
        self.fieldnames = fieldnames
        file_exists = os.path.isfile(filename) 
        self.f = open(filename, 'a', newline='')
        self.csv_writer = csv.DictWriter(self.f, fieldnames=fieldnames, delimiter=',')
        if not file_exists:
            self.csv_writer.writeheader()

    def addLine(self, in_dict):
        if len(in_dict.keys()) != len(self.fieldnames):
            print(in_dict.keys())
            print(in_dict)
            print(self.fieldnames)
        
        assert len(in_dict.keys()) == len(self.fieldnames)
        self.csv_writer.writerow(in_dict)
        self.f.flush()

    def close_csv(self):
        self.f.close()



class EnvSyn:
    def __init__(self, algo, noise, create_log=1, timestamp='', save_contexts=0, contexts_file='', ver=1):
        
        assert noise is not None
        self.noise_dist = noise['dist']
        self.noise_var = noise['var']
        self.create_log = create_log
        self.save_contexts = save_contexts
        self.algo = algo
        self.version = ver
        
        self.step_counter = 0
        min_action, max_action, action_dim = -2, 2, 1
        min_obs, max_obs = 0, 1
        self.obs_dim = 3 if self.version == 2 else 2
        
        
        self.action_space = spaces.Box(low=min_action, high=max_action, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, shape=(self.obs_dim,), dtype=np.float32)
        
        self.obs = self.observation_space.sample()
                
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S") if timestamp=='' else timestamp
        self.path = os.path.join(os.getcwd(), 'results', self.timestamp)
        
        if self.save_contexts:
            assert len(contexts_file) == 0
            fieldnames = ['step', 'context0', 'context1', 'context2']
            filename = os.path.join(self.path, self.timestamp+'_context_log.csv')
            os.makedirs(self.path, exist_ok=True)
            self.csv_h_contexts = csv_handler(filename, fieldnames)
            
            context2 = self.obs[2] if self.version == 2 else -1
            measure = {'step' : self.step_counter, 'context0' : self.obs[0], 'context1': self.obs[1], 'context2': context2}
            self.csv_h_contexts.addLine(measure)
            
        self.reset(contexts_file)

        
    def reset(self, contexts_file=''):
        
        if self.create_log:
            fieldnames = ['step', 'context', 'action', 'reward', 'constraint_1', 'constraint_2', 'alpha']
            filename = os.path.join(self.path, self.timestamp+f'_{self.algo}_log.csv')
            os.makedirs(self.path, exist_ok=True)
            self.csv_h = csv_handler(filename, fieldnames)
        
        
        if len(contexts_file) > 0:
            with open(contexts_file) as in_file:
                self.n_lines = sum(1 for _ in in_file)
            
            self.f = open(contexts_file)
            self.contexts_file = csv.reader(self.f, delimiter=',')
            next(self.contexts_file)
            line = next(self.contexts_file)
            self.obs = np.zeros(self.obs_dim)
            
            for i in range(self.obs_dim):
                self.obs[i] = float(line[i+1])
        else:
            self.contexts_file = None
    
    def close(self):
        if self.create_log:
            self.csv_h.close_csv()
        if self.save_contexts:
            self.csv_h_contexts.close_csv()
        if self.contexts_file is not None:
            self.f.close()

    
    def step(self, action, alpha=None):
        
        action = action[0]
        m1 = self.obs[0]*action**2 + self.obs[1]*action 
        m2 = self.obs[0]*action**2 - self.obs[1]*action 
        m3 = self.obs[0]*(action-self.obs[2])**2 - self.obs[1]*(action-self.obs[2]) if self.version == 2 else -1
        
        if self.noise_dist == 'norm':
            m1 += np.random.normal(loc=0.0, scale=self.noise_var)
            m2 += np.random.normal(loc=0.0, scale=self.noise_var)
            if self.version == 2:
                m3 += np.random.normal(loc=0.0, scale=self.noise_var)
            
        if self.create_log:
            measure = {'step' : self.step_counter, 'context' : self.obs, 'action': action, 'reward' : m1, 'constraint_1': m2, 'constraint_2': m3, 'alpha': alpha}
            self.csv_h.addLine(measure)
            
        if self.contexts_file is not None:
            if self.step_counter >= self.n_lines:
                raise Exception('Context file ended.')
            line = next(self.contexts_file)
            for i in range(self.obs_dim):
                self.obs[i] = float(line[i+1])
        else:
            self.obs = self.observation_space.sample()
        self.step_counter += 1
        
        if self.save_contexts:
            context2 = self.obs[2] if self.version == 2 else -1
            measure = {'step' : self.step_counter, 'context0' : self.obs[0], 'context1': self.obs[1], 'context2': context2}
            self.csv_h_contexts.addLine(measure)
        
        obs = np.expand_dims(self.obs, axis=0)
        reward = (m1, (m2, m3)) if self.version == 2 else (m1, m2)
        done = False
        info = {}
        return obs, reward, done, info
    
    
    
    def gen_dataset(self, n_points, actions_per_context, ver=2):
        if ver == 1:
            return self.gen_dataset_v1(n_points, actions_per_context)
        elif ver == 2:
            return self.gen_dataset_v2(n_points, actions_per_context)
        
    
    def gen_dataset_v1(self, n_points, actions_per_context):
        n_context = int(np.floor(n_points / actions_per_context))
        actual_points = int(n_context * actions_per_context)
        
        contexts_actions = np.zeros((actual_points, 3))
        m1 = np.zeros((actual_points, 1))
        m2 = np.zeros((actual_points, 1))
        
        idx = 0
        
        for i in range(n_context):
            obs = self.observation_space.sample()
            for j in range(actions_per_context):
                contexts_actions[idx, 0:2] = obs
                action = self.action_space.sample()
                contexts_actions[idx, 2] = action

                m1_i = obs[0]*action**2 + obs[1]*action 
                m2_i = obs[0]*action**2 - obs[1]*action 
                m1_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m2_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m1[idx] = m1_i
                m2[idx] = m2_i
                
                idx += 1
            
        return contexts_actions, m1, m2
    
    def gen_dataset_v2(self, n_points, actions_per_context):
        n_context = int(np.floor(n_points / actions_per_context))
        actual_points = int(n_context * actions_per_context)
        
        contexts_actions = np.zeros((actual_points, 4))
        m1 = np.zeros((actual_points, 1))
        m23 = np.zeros((actual_points, 2))
        
        idx = 0
        
        for i in range(n_context):
            obs = self.observation_space.sample()
            for j in range(actions_per_context):
                contexts_actions[idx, 0:3] = obs
                action = self.action_space.sample()
                contexts_actions[idx, 3] = action

                m1_i = obs[0]*action**2 + obs[1]*action 
                m2_i = obs[0]*action**2 - obs[1]*action 
                m3_i = obs[0]*(action-obs[2])**2 - obs[1]*(action-obs[2])
                
                m1_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m2_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m3_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m1[idx] = m1_i
                m23[idx,:] = [m2_i, m3_i]
                
                idx += 1
            
        return contexts_actions, m1, m23
