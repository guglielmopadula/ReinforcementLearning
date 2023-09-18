import numpy as np

## A miner has 5 mines in front of him and has to decide in which one to loot at any time point.

class KBandit():
    def __init__(self,starting_values,reward, K):
        self.K=K
        self.reward=reward
        self.starting_values=starting_values
        self.reward=reward
    
    def policy_update(self, reward, action, action_history, reward_history):
        return 0

    def alpha(self,t):
        return 1.

    def action_selection(self,reward,alpha, q, action_history, reward_history,t):
        return 0
    def train(self,t):
        opt_history=[]
        action_history=[]
        reward_history=[]
        q=self.starting_values
        a=self.action_selection(reward,1,q,action_history,reward_history,0)
        opt=np.argmax(self.reward(t))
        action_history=[a]
        r=self.reward(0)[a]
        reward_history=[r]
        opt_history=[opt]
        for i in range(1,t):
            q=self.policy_update(self.reward,1,a,action_history,reward_history,q,i)
            a=self.action_selection(reward,1,q,action_history,reward_history,i)
            opt=np.argmax(self.reward(t))
            r=self.reward(t)[a]
            reward_history.append(r)
            action_history.append(a)
            opt_history.append(opt)


        return action_history,reward_history,opt_history
    
        

class GreedySimpleAveragingKBandit(KBandit):

    def policy_update(self, reward, alpha, action, action_history, reward_history,q,t):
        q_new=q.copy()
        q_new[action]=q[action]+self.alpha(t)*(self.reward(t)[action]-q[action])
        return q_new
    
    def alpha(self,t):
        return 1/t

    def action_selection(self,reward, alpha, q, action_history, reward_history,t):
        return np.argmax(q)


def reward(t):
    return (2*np.ones(5)+2*np.arange(5)+0.01*np.random.rand(5))
starting_values=np.ones(5)

test=GreedySimpleAveragingKBandit(starting_values,reward,5)


class EpsSimpleAveragingKBandit(KBandit):
    def __init__(self, starting_values, reward, K):
        super().__init__(starting_values, reward, K)
        self.eps=0.1
    def policy_update(self, reward, alpha, action, action_history, reward_history,q,t):
        q_new=q.copy()
        q_new[action]=q[action]+self.alpha(t)*(self.reward(t)[action]-q[action])
        return q_new
    
    def action_selection(self,reward, alpha, q, action_history, reward_history,t):
        s=np.random.binomial(1,self.eps)
        if s:
            new_action=np.random.randint(self.K)
        else:
            new_action=np.argmax(q)
        return new_action

test=EpsSimpleAveragingKBandit(starting_values,reward,5)

action,rew,opt=test.train(100)



class EpsAlphaConstantKBandit(KBandit):
    def __init__(self, starting_values, reward, K):
        super().__init__(starting_values, reward, K)
        self.eps=0.1
    def policy_update(self, reward, alpha, action, action_history, reward_history,q,t):
        q_new=q.copy()
        q_new[action]=q[action]+self.alpha(t)*(self.reward(t)[action]-q[action])
        return q_new
    
    def action_selection(self,reward, alpha, q, action_history, reward_history,t):
        s=np.random.binomial(1,self.eps)
        if s:
            new_action=np.random.randint(self.K)
        else:
            new_action=np.argmax(q)
        return new_action

class UPBAlphaConstantKBandit(KBandit):
    def __init__(self, starting_values, reward, K):
        super().__init__(starting_values, reward, K)
        self.eps=0.1
    def policy_update(self, reward, alpha, action, action_history, reward_history,q,t):
        q_new=q.copy()
        q_new[action]=q[action]+self.alpha(t)*(self.reward(t)[action]-q[action])
        print(q_new)
        return q_new
    
    def compute_na(self,action):
        l=np.zeros(self.K)
        for i in range(self.K):
            tmp=np.array(action)
            l[i]=len(tmp[tmp==i])
        return l

    
    def action_selection(self,reward, alpha, q, action_history, reward_history,t):
        tmp=np.nan_to_num((1*np.sqrt(np.log(t)/self.compute_na(action_history))),nan=np.inf)
        new_action=np.argmax(q+tmp)
        return new_action

test=UPBAlphaConstantKBandit(starting_values,reward,5)

action,rew,opt=test.train(100)
print(np.cumsum((np.abs(np.array(action)-np.array(opt))==0).astype(int))/np.arange(1,101))
