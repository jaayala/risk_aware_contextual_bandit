import torch
import numpy as np

class constrained_loss:
    def __init__(self, constr_val, landa, distrib_critic, quantile_critic, constr_op='max', device='cpu'):
        if constr_op not in ('max', 'min'):
            raise Exception('Operation not allowed.')
        self.constr_op = constr_op
        self.device = device
        constr_val = constr_val if hasattr(constr_val, "__len__") else [constr_val]
        self.c_val = torch.FloatTensor(constr_val).to(self.device)
        self.distrib_critic = distrib_critic
        self.zero = torch.FloatTensor([0]).to(self.device)
        self.quantile_critic = np.array(quantile_critic)
        self.n_constraints = len(self.c_val)
        
        if not hasattr(landa, "__len__"):
            self.landa = np.ones(self.n_constraints) * landa
        assert len(self.landa) == self.n_constraints
       
    def compute_loss(self, actor, critics, states, alpha, alpha_tensor):
        
        if len(critics) == 1:
            policy_loss = critics[0].forward(states, actor.forward(states)).mean()
            return policy_loss
        else:
            batch_size = states.shape[0]
            penalty = torch.zeros(batch_size, 1).to(self.device)
                    
            for i in range(self.n_constraints):

                if self.distrib_critic:      
                    pos = np.where(self.quantile_critic==alpha)[0]
                    assert len(pos) == 1
                    actor_input = [states, alpha_tensor]
                    c = critics[i+1].forward(states, actor.forward(*actor_input))[:, pos]
                else:
                    actor_input = [states]
                    c = critics[i+1].forward(states, actor.forward(*actor_input)).mean(1)
               
                penalty_aux = self.c_val[i] - c  if self.constr_op=='min' else c - self.c_val[i]
                penalty_aux = torch.maximum(self.zero, penalty_aux)
                penalty_aux = torch.mul(penalty_aux, self.landa[i])
                penalty += penalty_aux.reshape(-1,1)
                
            expected_cost = critics[0].forward(states, actor.forward(*actor_input)).mean(1).reshape(-1,1)
            policy_loss = expected_cost + penalty
            return policy_loss.mean()
    
    
    
class critic_objective:
    def __init__(self, n_critics, landa=0, constr_val=0, constr_op='max', device='cpu'):
        self.n_critics = n_critics
        self.device = device
        if self.n_critics == 1:
            self.landa = landa
            self.zero = torch.FloatTensor([0]).to(self.device)
            self.constr_op = constr_op
            constr_val = constr_val if hasattr(constr_val, "__len__") else [constr_val]
            self.c_val = torch.FloatTensor(constr_val).to(self.device)
            self.n_constraints = len(self.c_val)
            
            if not hasattr(landa, "__len__"):
                self.landa = np.ones(self.n_constraints) * landa
            assert len(self.landa) == self.n_constraints
            
    def get_objective(self, reward, constraint, idx):
                
        if self.n_critics > 1:
            out = reward if idx == 0 else constraint[:,idx-1].reshape(-1,1)
        elif self.n_critics == 1:
            assert self.n_constraints == constraint.shape[1]
            batch_size = constraint.shape[0]

            penalty = torch.zeros(batch_size, 1).to(self.device)
            for i in range(self.n_constraints):
                penalty_aux = self.c_val[i] - constraint[:,i]  if self.constr_op=='min' else constraint[:,i] - self.c_val[i]
                penalty_aux = torch.maximum(self.zero, penalty_aux)
                penalty_aux = torch.mul(penalty_aux, self.landa[i])
                penalty += penalty_aux.reshape(-1,1)
            out = reward + penalty
        return out
    
    

