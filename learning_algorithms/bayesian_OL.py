import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from constrained_bayes_opt import ConstrainedBayesOpt


class Bayesian_OL:
    def __init__(self, env, beta_val=2.5, contexts={'c1': '', 'c2': ''}, actions={'a': np.linspace(0, 1, 1001)}, constr_val=0.3, constr_greater=0, 
                 points=[], rewards=[], constraint_vals=[]):
        self.env = env
        self.actions = actions
        self.contexts = contexts
        self.beta_val = beta_val
        
        self.dim_states = self.env.observation_space.shape[0]
        self.dim_actions = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.last_state = None
        # self.safeset_size = []
        
        action_dim = len(self.actions)
        context_dim = len(self.contexts)

        nConstraints = len(constr_val) if hasattr(constr_val, "__len__") else 1
        length_scale_list = [np.ones(context_dim+action_dim) for _ in range(nConstraints+1)]
        kernels = [WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=length_scale_list[i]) for i in range(len(length_scale_list))]
        
        
        assert hasattr(constr_val, "__len__") == hasattr(constr_greater, "__len__")
        constraints_thres = np.array(constr_val) if hasattr(constr_val, "__len__") else np.array([constr_val])
        # Maximum value that the constraint can take
        # if zero; the contrained function should take values *lower than* the threshold specified above
        constr_greater = np.array(constr_greater) if hasattr(constr_greater, "__len__") else np.array([constr_greater])

        optimizedKernels = 0
        
        safeset = np.array([[0.4]])
            
        self.optimizer = ConstrainedBayesOpt(all_actions_dict=self.actions, contexts=self.contexts, kernels=kernels, 
                             constraint_thres=constraints_thres, constr_greater=constr_greater, safeset=safeset, 
                             optimizedKernels=optimizedKernels, beta_val=self.beta_val, points=points, rewards=rewards, constraint_vals=constraint_vals, 
                             init_random=30)
        

        
    def denormalize_action(self, action):
        action = np.minimum(action, 1)
        action = np.maximum(action, 0)
        denorm_act = action * (self.action_high - self.action_low) + self.action_low
        return denorm_act
    
    def get_action(self, state):
        context = self.optimizer.array_to_context(state[0])
        action, empty_safeset = self.optimizer.suggest(context)
        return action, empty_safeset
                    

    def run(self, n_iters, evaluation=0, alpha_exec=None):

        if self.last_state is None:
            initial_action = self.env.action_space.sample()
            state, _, _, info = self.env.step(initial_action)
        else:
            state = self.last_state
            info = self.last_info

        for step in range(n_iters-1):
            if (step+1) % 50 == 0:
                print(f'Step {step+1} / {n_iters}')
                
            action, empty_safeset = self.get_action(state)
            denorm_action = self.denormalize_action(self.optimizer.action_to_array(action))
            
            new_state, metrics, done, info = self.env.step(denorm_action, alpha=empty_safeset)
            reward, constraint =  metrics
            state = new_state
            if evaluation == 0:
                self.optimizer.register(self.optimizer.array_to_context(state[0]), action, reward, np.array([constraint]))
                
            if done:
                break
        self.last_state = state
        self.last_info = info


    def close(self):
        self.env.close()



