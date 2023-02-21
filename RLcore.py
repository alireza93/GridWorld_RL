import numpy as np

class MDP():
    def __init__(self, nStates, actions, rewards, terminal):
        self.actions = np.array(actions)
        self.states = np.array(range(nStates))
        self.rewards = np.array(rewards)
        self.terminal = np.array(terminal)
    def dynamics(self,sp,r,s,a):
        return 0;
    def stateDiagram(self):
        ...


class DPRL():
    def __init__(self, mdp, discount):
        self.mdp = mdp
        self.discount = discount
        self.stateValue = np.zeros(mdp.states.shape)
        self.policy = np.zeros([len(mdp.states),len(mdp.actions)])
        self.policy.fill(1/len(self.mdp.actions))
        self.policy[self.mdp.terminal,:] = 0
    def policyIteration(self):
        while True:
            self.policyEvaluation()
            stable = self.policyImprovement()
            if stable:
                return
    def policyEvaluation(self):
        theta = 0.1        
        while True:
            delta = 0
            for s in self.mdp.states:
                v = self.stateValue.copy()
                self.stateValue[s] = sum(self.policy[s,i_a]*
                                sum((r+self.discount*self.stateValue[sp])*
                                self.mdp.dynamics(sp,r,s,self.mdp.actions[i_a]) 
                                for sp in self.mdp.states for r in self.mdp.rewards) 
                                for i_a in range(len(self.mdp.actions)))
                delta = max(delta,np.linalg.norm(v-self.stateValue))
            if delta<theta:
                break;
    def policyImprovement(self):
        policyStable = True
        for s in self.mdp.states:
            if s in self.mdp.terminal:
                continue
            oldActions = self.policy[s,:].copy(); #random.choices(actions, weights=policy[s,:], k=1)
            actionValues = np.array(list(map(lambda i_a: 
                    sum((r+self.discount*self.stateValue[sp])*self.mdp.dynamics(sp,r,s,self.mdp.actions[i_a])
                    for sp in self.mdp.states for r in self.mdp.rewards), range(len(self.mdp.actions)))))
            greedyActions = np.zeros(self.mdp.actions.shape)
            maxIdx = np.argwhere(actionValues==actionValues.max()).flatten()
            greedyActions[maxIdx] = 1/maxIdx.shape[0]
            self.policy[s,:] = greedyActions
            if (oldActions!=greedyActions).any():
                policyStable = False
        self.policy[self.mdp.terminal,:] = 0
        return policyStable;