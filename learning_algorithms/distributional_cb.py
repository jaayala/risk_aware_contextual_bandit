import numpy as np
import torch
import torch.autograd
import torch.optim as optim
import random
import os

from utils_learning import noNoise
from utils_learning import Memory_constriant as Memory
from utils_learning import quantile_huber_loss

from models_v2 import MLP


class DistributionalCBBase:
    def __init__(self, env, noise, batch_size, actor_loss_function, critic_objective_class, landa, A_train, A_explo, alpha_input_actor,
                     save_model, updates_per_step, device):
        self.env = env
        if noise is not None:
            self.noise = noise
        else:
            self.noise = noNoise()
        self.batch_size = batch_size
        self.actor_loss_function = actor_loss_function
        self.critic_objective = critic_objective_class
        self.device = device
        self.landa = torch.FloatTensor(np.array([landa])).to(self.device)
        
        self.alpha_exec = None
        self.A_train = A_train
        self.A_explo = A_explo
        self.A_explo_tensor = []
        for a in A_explo:
            self.A_explo_tensor.append(torch.FloatTensor(np.array([a])).to(self.device))
        
        self.alpha_input_actor = alpha_input_actor # if 1, we input landa to actor
        self.save = save_model
        self.updates_per_step = updates_per_step
        self.dim_states = self.env.observation_space.shape[0]
        self.dim_actions = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.save_path = self.env.path
        
        self.last_state = None
        self.last_info = None
        self.noise.reset()

    def denormalize_action(self, action):
        action = np.minimum(action, 1)
        action = np.maximum(action, 0)
        denorm_act = action * (self.action_high - self.action_low) + self.action_low
        return denorm_act
    
    def get_action(self, state):
        state = state[0]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.alpha_input_actor:
            if self.evaluation:
                alpha = self.alpha_exec
            else:
                alpha = random.choice(self.A_explo_tensor)
            action = self.actor.forward(state, alpha.unsqueeze(0))
            alpha = alpha.detach().cpu().numpy()[0]
        else:
            action = self.actor.forward(state)
            alpha = None
        action = action.detach().cpu().numpy()[0]
        return action, alpha
            

    def run(self, n_iters, evaluation=0, alpha_exec=None):
        
        if alpha_exec is not None:
            self.alpha_exec = torch.FloatTensor(np.array([alpha_exec])).to(self.device)
        
        self.evaluation = evaluation
        for mod in self.learning_modules:
            if evaluation == 0:
                mod.train()
            else:
                mod.eval()
                assert self.alpha_exec is not None
                
                
        if self.last_state is None:
            initial_action = self.env.action_space.sample()
            state, _, _, info = self.env.step(initial_action)
        else:
            state = self.last_state
            info = self.last_info

        for step in range(n_iters-1):
            if (step+1) % 100 == 0:
                print(f'Step {step+1} / {n_iters}')
                
            action, alpha = self.get_action(state)
            if evaluation == 0:
                noise = self.noise.get_noise()
                if len(noise) == len(action):
                    action += noise
                elif len(noise) == 1:
                    action += noise[0]
                else:
                    raise Exception(f'Generated noise with length {len(noise)} and action with length {len(action)}')

            denorm_action = self.denormalize_action(action)
            
            new_state, metrics, done, info = self.env.step(denorm_action, alpha=alpha)
            reward, constraint =  metrics
            # print(f'action: {denorm_action}; reward: {reward}; constraint: {constraint}')
           
            if evaluation == 0:
                self.memory.push(state[0], action, reward, constraint)
                
                if len(self.memory) > self.batch_size:
                    for _ in range(self.updates_per_step):
                        self.update()
            state = new_state
            
            if done:
                break
        self.last_state = state
        self.last_info = info


    def save_models(self):
        torch.save(self.actor.state_dict(), os.path.join(self.save_path, 'actor_model'))
        torch.save(self.critic.state_dict(), os.path.join(self.save_path, 'critic_model'))

    def close(self):
        self.env.close()
        if self.save:
            self.save_models()



class DistributionalCB(DistributionalCBBase):
    def __init__(self, env, noise, batch_size, actor_kwargs, critics_kwargs, actor_loss_function, critic_loss, critic_objective_class, A_train, A_explo,
                 convnet_kwargs=None, landa=2.5, alpha_input_actor=1, actor_learning_rate=1e-4, critics_learning_rate=1e-3, 
                 max_memory_size=2000, save_model=0, updates_per_step=1, device='cpu'):
        super().__init__(env, noise, batch_size, actor_loss_function, critic_objective_class, landa, A_train, A_explo, alpha_input_actor,
                         save_model, updates_per_step, device)
        
        if len(critics_kwargs) == 1:
            assert alpha_input_actor == 0
        
        self.critics_loss_kwargs = []
        for critic_kwargs in critics_kwargs:
            if 'quantile_vals' in critic_kwargs:
                quantile_vals = critic_kwargs.pop('quantile_vals')
                
                if quantile_vals is not None:
                    assert isinstance(quantile_vals, list)
                    assert 'n_quantiles' in critic_kwargs
                    assert len(quantile_vals) == critic_kwargs['n_quantiles']
                    quantile_vals = torch.FloatTensor(quantile_vals).to(device)
                self.critics_loss_kwargs.append({'cum_prob' : quantile_vals})
            else:
                self.critics_loss_kwargs.append({})
                
        if convnet_kwargs is None:
            actor_kwargs.update({'in_features' : self.dim_states + self.alpha_input_actor, 'out_features' : self.dim_actions, 'device' : device})
            self.actor = MLP(**actor_kwargs)
            
            self.critics = []
            critic_input_size = self.dim_states + self.dim_actions

            for critic_kwargs in critics_kwargs:
                out_features = critic_kwargs.pop('n_quantiles') if 'n_quantiles' in critic_kwargs else 1
                critic_kwargs.update({'in_features' : critic_input_size, 'out_features' : out_features, 'device' : device})
                self.critics.append(MLP(**critic_kwargs))
        else:
            raise Exception('Not implemented.')
        
        self.learning_modules = [self.actor] + self.critics
        for m in self.learning_modules:
            print(m)
        
        # Training
        self.memory = Memory(max_memory_size)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        
        
        self.critics_criterion = []
        self.critics_optimizer = []
        for i, critic_kwargs in enumerate(critics_kwargs):
            
            loss = critic_loss[i] if type(critic_loss) is list else critic_loss

            if loss == 'quantile_huber_loss':
                self.critics_criterion.append(quantile_huber_loss)
            elif loss == 'MSELoss':
                self.critics_criterion.append(torch.nn.MSELoss())
            
            critic_lr = critics_learning_rate[i] if hasattr(critics_learning_rate, '__len__') else critics_learning_rate
            self.critics_optimizer.append(optim.Adam(self.critics[i].parameters(), lr=critic_lr))
           
    
    def update(self):
        states, actions, rewards, constraints = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        constraints = torch.FloatTensor(np.array(constraints)).to(self.device)
        
        # Critics loss
        for i, critic in enumerate(self.critics):
            
            Qvals = critic.forward(states, actions)
            obj = self.critic_objective.get_objective(rewards, constraints, i)
            
            critic_loss = self.critics_criterion[i](Qvals, obj, **self.critics_loss_kwargs[i])
            
            self.critics_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critics_optimizer[i].step()
                   
        # policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean() # maximize
        # policy_loss = self.critic_1.forward(states, self.actor.forward(states), actor_mf_actions_hist).mean() +   # minimize

        for alpha in self.A_train:
            alpha_tensor = torch.FloatTensor(np.ones((self.batch_size, 1))*alpha).to(self.device)
            policy_loss =  self.actor_loss_function.compute_loss(self.actor, self.critics, states, alpha, alpha_tensor)
  
            # update networks
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            



