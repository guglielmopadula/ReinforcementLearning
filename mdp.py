import numpy as np
from queue import PriorityQueue

def continous_example():
    ###Example 3.3 of surton and burton

    #Ordering

    #battery state:low,high
    #action:wait,search,recharge
    #reward: -3,0,wait,search    
    alpha=0.5
    beta=0.5
    r_wait=1
    r_search=2
    gamma=0.9
    p=np.zeros((2,4,2,3))
    p[1,3,1,1]=alpha
    p[0,3,1,1]=1-alpha
    p[1,0,0,1]=1-beta
    p[0,3,0,1]=beta
    p[1,2,1,0]=1
    p[0,2,0,0]=1
    p[1,1,0,2]=1
    r=np.array([-3,0,r_wait,r_search])
    return r,p


class MDP():
    def __init__(self,pmatrix, rewards):
        self.pmatrix=pmatrix
        self.num_states=pmatrix.shape[0]
        self.num_actions=pmatrix.shape[3]
        self.num_rewards=rewards.shape[0]
        self.rewards=rewards
    
    def _decode(self,x):
        return x//self.num_actions,x%self.num_actions

    def _encode(self,x,y):
        return x*self.num_actions+y

    def random_policy(self):
        policy=np.random.rand(self.num_states,self.num_actions)
        for i in range(self.num_actions):
            for j in range(self.num_states):
                if np.sum(self.pmatrix[:,:,j,i])==0:
                    policy[j,i]=0

        for i in range(self.num_states):
            policy[i,:]=policy[i,:]/np.sum(policy[i,:])
        return policy
    
    def create_model(self):
        model=np.zeros((self.num_states,self.num_actions,2))
        for i in range(self.num_states):
            for j in range(self.num_actions):
                tmp=self.pmatrix[:,:,i,j].reshape(-1)
                index=np.argmax(tmp)
                index0=index//self.num_rewards
                index1=index%self.num_rewards
                model[i,j,0]=index0
                model[i,j,1]=self.rewards[index1]
        return model

    def random_q_function(self):
        tmp=self.random_policy()
        tmp=tmp*np.random.rand(*tmp.shape)
        return tmp

    def _step(self,s,a):
        s_temp,r_temp=self._choice_bivariate(self.pmatrix[:,:,s,a])
        return s_temp,r_temp
    
    def _choice_bivariate(self,p):
        m=p.shape[0]
        n=p.shape[1]
        x=np.random.choice(m*n,p=p.reshape(-1))
        return x//n, x%n




class knownMDP(MDP):

    # policy here is (s,a) even if it represents a|s for speeding up computations; it also works if p is not known, but on only samples are avaiable
    def evaluate_policy(self,policy,epsilon,gamma):
        v=np.zeros(self.num_states)
        diff=np.inf
        while diff>epsilon:
            v_old=v.copy()
            v=np.sum(policy*np.sum(self.pmatrix*(self.rewards.reshape(1,self.rewards.shape[0],1,1)+gamma*v_old.reshape(v_old.shape[0],1,1,1)),axis=(0,1)).reshape(self.num_states,self.num_actions),axis=1)
            diff=np.max(np.abs(v-v_old))

        return v
    
    #policy here is deterministic
    def optimal_policy_value_iteration(self,gamma,epsilon):
        v=np.random.rand(self.num_states)
        diff=np.inf
        while diff>epsilon:
            v_old=v.copy()
            v=np.max(np.sum(self.pmatrix*(self.rewards.reshape(1,self.rewards.shape[0],1,1)+gamma*v_old.reshape(v_old.shape[0],1,1,1)),axis=(0,1)).reshape(self.num_states,self.num_actions),axis=1)
            diff=np.max(np.abs(v-v_old))
        return np.argmax(np.sum(self.pmatrix*(self.rewards.reshape(1,self.rewards.shape[0],1,1)+gamma*v_old.reshape(v_old.shape[0],1,1,1)),axis=(0,1)).reshape(self.num_states,self.num_actions),axis=1)
    


class ModelFreeEpisodicMDP(MDP):
        #mcmc methods are very slow in both speed and convergence
    # by convention, the terminal states are always the last ones
    def __init__(self, pmatrix, rewards, num_terminal_states):
        super().__init__(pmatrix, rewards)
        self.num_terminal_states=num_terminal_states
        self.terminal_states=np.arange(self.num_states-num_terminal_states,self.num_states).tolist()
    
    def _generate_episode(self,policy):
        s=[]
        a=[]
        r=[]
        s.append(np.random.randint(self.num_states-self.num_terminal_states))
        a.append(np.random.choice(self.num_actions,p=policy[s[0],:]))
        flag=1
        t=0
        while flag:
            s_temp,r_temp=self._choice_bivariate(self.pmatrix[:,:,s[t],a[t]])
            t=t+1
            s.append(s_temp)
            r.append(r_temp)
            a_temp=np.random.choice(self.num_actions,p=policy[s_temp,:])
            a.append(a_temp)
            flag=self.terminal_states.count(s[t])==0
        return np.array(s),np.array(r),np.array(a)



    
    def evaluate_policy_mcmc(self,policy,gamma,num_ep):
        returns=[[] for _ in range(self.num_states)]
        V=np.zeros(self.num_states)
        for _ in range(num_ep): 
            s,r,a=self._generate_episode(policy)
            G=0
            for t in range(len(r)-2,-1,-1):
                G=gamma*G+r[t+1]
                if not np.sum(s[:t]==s[t]).astype(int):
                    returns[s[t]].append(G)
                    V[s[t]]=np.mean(np.array(returns[s[t]]))
        return V
    
    def onpolicy_mcmc_optimal(self,epsilon,gamma,num_ep):
        policy=self.random_policy()
        q=self.random_q_function()
        returns=[[] for i in range(self.num_actions*self.num_actions)]
        
        for i in range(num_ep):
            G=0
            s,r,a=self._generate_episode(policy)
            coupled=self._encode(s,a)
            for t in range(len(r)-2,-1,-1):
                G=gamma*G+r[t+1]
                if not np.sum(coupled[:t]==coupled[t]).astype(int):
                    returns[coupled[t]].append(G)
                    s_temp,a_temp=self._decode(coupled[t])
                    q[s_temp,a_temp]=np.mean(returns[coupled[t]])
                    aopt=np.argmax(q[s_temp,:])
                    l=np.sum(policy[s[t],:]>0)
                    for a in range(self.num_actions):
                        if policy[s[t],a]>0:
                            if a!=aopt:
                                policy[s[t],a]=epsilon/l
                            else:
                                policy[s[t],a]=1-epsilon+epsilon/l
        return policy


    def evaluate_q_off_mcmc(self,policy,gamma,num_ep):
        q=self.random_q_function()
        C=np.zeros(q.shape)
        for _ in range(num_ep): 
            G=0
            W=1
            b=self.random_policy()
            s,r,a=self._generate_episode(b)
            for t in range(len(r)-2,-1,-1):
                G=gamma*G+r[t+1]
                C[s[t],a[t]]=C[s[t],a[t]]+W
                q[s[t],a[t]]=q[s[t],a[t]]+W/C[s[t],a[t]]*(G-q[s[t],a[t]])
                W=W*policy[s[t],a[t]]/b[s[t],a[t]]
        return q
    
    def offpolicy_mcmc_optimal(self,gamma,num_ep):
        q=self.random_q_function()
        policy=np.argmax(q,axis=1)
        C=np.zeros(q.shape)
        for _ in range(num_ep): 
            G=0
            W=1
            b=self.random_policy()
            s,r,a=self._generate_episode(b)
            for t in range(len(r)-2,-1,-1):
                G=gamma*G+r[t+1]
                C[s[t],a[t]]=C[s[t],a[t]]+W
                q[s[t],a[t]]=q[s[t],a[t]]+W/C[s[t],a[t]]*(G-q[s[t],a[t]])
                policy[s[t]]=np.argmax(q[s[t]])
                if policy[s[t]]!=a[t]:
                    continue
                W=W*1/b[s[t],a[t]]
        return policy

    def evaluate_policy_td0(self, policy, gamma,alpha,n_ep):
        v=np.zeros(self.num_states)
        for i in range(n_ep):
            s,r,a=self._generate_episode(policy)
            for i in range(len(s)-2):
                v[s[i]]=v[s[i]]+alpha*(r[i+1]+gamma*v[s[i+1]]-v[s[i]])
        return v

    def evaluate_policy_tdn(self, policy, gamma,alpha,n,n_ep):
        v=np.zeros(self.num_states)
        for _ in range(n_ep):
            s,r,a=self._generate_episode(policy)
            for t in range(len(s)-1):
                tau=t-n+1
                if tau>=0:
                    G=np.sum(np.array([gamma**(i-tau-1)*r[i-1] for i in range(tau+1,tau+n+1)]))
                    if tau+n<len(s):
                        G=G+gamma**(n)*v[s[tau+n]]
                    v[s[tau]]=v[s[tau]]+alpha*(G-v[s[tau]])
        return v


    def generate_greedy_policy_from_q(self,q,epsilon):
        policy=q.copy()
        for i in range(self.num_states):
            l=np.sum(q[i]>0)
            aopt=np.argmax(q[i])
            for j in range(self.num_actions):
                if q[i,j]>0:
                    if j==aopt:
                        policy[i,j]=1-epsilon+epsilon/l
                    else:
                        policy[i,j]=epsilon/l
        return policy



    def sarsa_optimal_q(self,gamma,alpha,n_ep):
        q=self.random_q_function()
        for _ in range(n_ep):
            policy=self.generate_greedy_policy_from_q(q,0.10)
            s,r,a=self._generate_episode(policy)
            for t in range(len(s)-2):
                policy=self.generate_greedy_policy_from_q(q,0.10)
                aprime=np.random.choice(self.num_actions,p=policy[s[t+1]])
                q[s[t],a[t]]=q[s[t],a[t]]+alpha*(r[t+1]+gamma*q[s[t+1],aprime]-q[s[t],a[t]])
        return q 
    
    def sarsa_optimal_q_n(self, gamma,alpha,n,n_ep):
        v=np.zeros(self.num_states)
        q=self.random_q_function()
        for _ in range(n_ep):
            policy=self.generate_greedy_policy_from_q(q,0.10)
            s,r,a=self._generate_episode(policy)
            a_new=a.copy()
            for t in range(len(s)-1):
                tau=t-n+1
                aprime=np.random.choice(self.num_actions,p=policy[s[t+1]])
                a_new[t+1]=aprime
                if tau>=0:
                    G=np.sum(np.array([gamma**(i-tau-1)*r[i-1] for i in range(tau+1,tau+n+1)]))
                    if tau+n<len(s):
                        G=G+gamma**(n)*q[s[tau+n],a[tau+n]]
                    q[s[tau],a_new[tau]]=q[s[tau],a_new[tau]]+alpha*(G-q[s[tau],a_new[tau]])
                    policy=self.generate_greedy_policy_from_q(q,0.10)
        return q

    def off_sarsa_optimal_q_n(self, gamma,alpha,n,b,n_ep):
        v=np.zeros(self.num_states)
        q=self.random_q_function()
        for _ in range(n_ep):
            policy=self.generate_greedy_policy_from_q(q,0.10)
            s,r,a=self._generate_episode(policy)
            a_new=a.copy()
            for t in range(len(s)-1):
                tau=t-n+1
                aprime=np.random.choice(self.num_actions,p=b[s[t+1]])
                a_new[t+1]=aprime
                if tau>=0:
                    G=np.sum(np.array([gamma**(i-tau-1)*r[i-1] for i in range(tau+1,tau+n+1)]))
                    rho=np.prod(np.array([policy[s[i],a[i]]/b[s[i],a[i]] for i in range(tau+1,tau+n+1)]))
                    if tau+n<len(s):
                        G=G+gamma**(n)*q[s[tau+n],a[tau+n]]
                    q[s[tau],a_new[tau]]=q[s[tau],a_new[tau]]+rho*alpha*(G-q[s[tau],a_new[tau]])
                    policy=self.generate_greedy_policy_from_q(q,0.10)
        return q

    def sigma_optimal_q_n(self, gamma,alpha,n,b,n_ep):
        v=np.zeros(self.num_states)
        q=self.random_q_function()
        for _ in range(n_ep):
            policy=self.generate_greedy_policy_from_q(q,0.10)
            s,r,a=self._generate_episode(policy)
            sigma=np.random.rand(len(s))
            rho=np.zeros(len(s))
            a_new=a.copy()
            for t in range(len(s)-1):
                tau=t-n+1
                aprime=np.random.choice(self.num_actions,p=policy[s[t+1]])
                rho[t]=np.array(policy[s[t],a[t]]/b[s[t],a[t]])
                a_new[t+1]=aprime
                if tau>=0:
                    G=0
                    for k in range(t,tau,-1):
                        V=np.sum([policy[s[k],a]*q[s[k],a] for a in range(self.num_actions)])
                        G=r[k-1]+gamma*(sigma[k]*rho[k]+(1-sigma[k]*policy[s[k],a[k]]))*(G-q[s[k],a[k]])+gamma*V
                    q[s[tau],a_new[tau]]=q[s[tau],a_new[tau]]+alpha*(G-q[s[tau],a_new[tau]])
                    policy=self.generate_greedy_policy_from_q(q,0.10)
        return q


    def tree_optimal_q_n(self, gamma,alpha,n,n_ep):
        v=np.zeros(self.num_states)
        q=self.random_q_function()
        for _ in range(n_ep):
            policy=self.generate_greedy_policy_from_q(q,0.10)
            s,r,a=self._generate_episode(policy)
            a_new=a.copy()
            for t in range(len(s)-1):
                tau=t-n+1
                aprime=np.random.choice(self.num_actions,p=policy[s[t+1]])
                a_new[t+1]=aprime
                if tau>=0:
                    G=r[t-1]+gamma*np.sum(policy[s[t],:]*q[s[t],:])
                    for k in range(t,tau,-1):
                        G=np.sum([policy[s[k],a]*q[s[k],a] for a in range(self.num_actions) if a!=a[k]])+gamma*policy[s[k],a[k]]*q[s[k],a[k]]*G
                    q[s[tau],a_new[tau]]=q[s[tau],a_new[tau]]+alpha*(G-q[s[tau],a_new[tau]])
                    policy=self.generate_greedy_policy_from_q(q,0.10)
        return q


    def q_learning(self,gamma,alpha,n_ep):
        q=self.random_q_function()
        for _ in range(n_ep):
            policy=self.generate_greedy_policy_from_q(q,0.10)
            s,r,a=self._generate_episode(policy)
            s=s[0]
            r=r[0]
            a=a[0]
            flag=True
            while flag:
                policy=self.generate_greedy_policy_from_q(q,0.10)
                aprime=np.random.choice(self.num_actions,p=policy[s,:])
                s_temp,r_temp=self._choice_bivariate(self.pmatrix[:,:,s,aprime])
                q[s,aprime]=q[s,aprime]+alpha*(r_temp+gamma*np.max(q[s_temp,:])-q[s,aprime])
                s=s_temp
                flag=self.terminal_states.count(s)==0
        policy=np.argmax(q,axis=1)
        return policy

    def double_q_learning(self,gamma,alpha,n_ep):
        q1=self.random_q_function()
        q2=self.random_q_function()
        for _ in range(n_ep):
            policy=self.generate_greedy_policy_from_q(q1+q2,0.10)
            s,r,a=self._generate_episode(policy)
            s=s[0]
            r=r[0]
            a=a[0]
            flag=True
            while flag:
                policy=self.generate_greedy_policy_from_q(q1+q2,0.10)
                aprime=np.random.choice(self.num_actions,p=policy[s,:])
                s_temp,r_temp=self._choice_bivariate(self.pmatrix[:,:,s,aprime])
                tmp=np.random.binomial(1,0.5)
                if tmp:
                    q1[s,aprime]=q1[s,aprime]+alpha*(r_temp+gamma*q2[s_temp,np.argmax(q1[s_temp,:])]-q1[s,aprime])
                else:
                    q2[s,aprime]=q2[s,aprime]+alpha*(r_temp+gamma*q1[s_temp,np.argmax(q2[s_temp,:])]-q2[s,aprime])

                s=s_temp
                flag=self.terminal_states.count(s)==0
        policy=np.argmax(q1+q2,axis=1)
        return policy

class ModelMDP(MDP):
    def __init__(self, pmatrix, rewards,model=None):
        super().__init__(pmatrix, rewards)
        if model is None:
            self.model=self.create_model()

    def one_step_tab_q_planning(self,alpha,gamma):
        q=self.random_q_function()
        for i in range(1000):
            policy=self.random_policy()
            s=np.random.randint(self.num_states)
            a=np.random.choice(self.num_actions,p=policy[s,:])
            sprime,r=self.model[s,a]
            sprime=sprime.astype(int)
            q[s,a]=q[s,a]+alpha*(r+gamma*np.max(q[sprime]-q[sprime,a]))
        return q
            
    def dyna_q(self,alpha,gamma):
        q=self.random_q_function()
        model=np.zeros_like(self.model)
        s=np.random.randint(self.num_states)
        for i in range(1000):
            a=np.argmax(q[s])
            s_next,r_next=self._step(s,a)
            q[s,a]=q[s,a]+alpha*(r_next+gamma*np.max(q[s_next])-q[s,a])
            model[s,a,0]=s_next
            model[s,a,1]=r_next
            for j in range(10):
                s=int(s_next)
                a=np.random.choice(self.num_actions,p=self.random_policy()[s])
                s_next,r_next=model[s,a]
                q[s,a]=q[s,a]+alpha*(r_next+gamma*np.max(q[int(s_next)])-q[s,a])
        return q,model
    
    def prioritized_sweeping(self,alpha,gamma):
        q=self.random_q_function()
        model=np.zeros_like(self.model)
        s=np.random.randint(self.num_states)
        pqueue=PriorityQueue()
        for i in range(1000):
            a=np.argmax(q[s])
            s_next,r_next=self._step(s,a)
            q[s,a]=q[s,a]+alpha*(r_next+gamma*np.max(q[s_next])-q[s,a])
            model[s,a,0]=s_next
            model[s,a,1]=r_next
            P=np.abs(r_next+gamma*np.max(q[s_next])-q[s,a])
            if P>0.001:
                pqueue.put([P,[s,a]])
            for j in range(10):
                if pqueue.empty():
                    continue
                tmp=pqueue.get()
                _,d=tmp
                s,a=d
                s_next,r_next=model[s,a]
                s_next=int(s_next)
                q[s,a]=q[s,a]+alpha*(r_next+gamma*np.max(q[int(s_next)])-q[s,a])
                for sbar in range(self.num_states):
                    for abar in range(self.num_actions):
                        if np.sum(self.pmatrix[:,:,sbar,abar])>0:
                            if model[sbar,abar,0]==s:
                                rbar=model[sbar,abar,1]
                                P=np.max(rbar+gamma*gamma*np.max(q[s])-q[sbar,abar])
                                if P>0.001:
                                    pqueue.put([P,[sbar,abar]])
        return q,model

class ApproxMDP(MDP):
    def v_fun(self):
        pass

    def v_grad(self):
        pass

    def q_fun(self):
        pass

    def q_grad(self):
        pass
    
    def q_fun_s(self,s,w):
        tmp=np.zeros(self.num_actions)
        for a in range(self.num_actions):
            if np.sum(self.pmatrix[:,:,s,a])>0:
                tmp[a]=self.q_fun(s,a,w)
        return tmp
    
    def pi_fun_s(self,s,theta):
        tmp=np.zeros(self.num_actions)
        for a in range(self.num_actions):
            if np.sum(self.pmatrix[:,:,s,a])>0:
                tmp[a]=self.pi_fun(s,a,theta)
        return tmp


    def pi_fun(self,s,a,theta):
        pass

    def pi_grad(self,s,a,theta):
        pass

    def pi_grad_log(self,s,a,theta):
        if np.sum(self.pmatrix[:,:,s,a])>0:
            return self.pi_grad(s,a,theta)/self.pi_fun(s,a,theta)
        else:
            return 0

    def semigradientsarsa(self,alpha,beta):
        rbar=0
        w=np.random.rand(self.num_actions*self.num_states)
        s=0
        a=np.argmax(self.q_fun_s(s,w))
        for i in range(100):
            s_next,r_next=self._step(s,a)
            a_next=np.argmax(self.q_fun_s(s_next,w))
            delta=r_next-rbar+self.q_fun(s_next,a_next,w)-self.q_fun(s,a,w)
            rbar=rbar+beta*delta
            w=w+alpha*delta*self.q_grad(s,a,w)
            s=s_next
            a=a_next
        return w

    def semigradientsarsa_n(self,alpha,beta,n):
        rbar=0
        s_vec=[]
        a_vec=[]
        r_vec=[]
        w=np.random.rand(self.num_actions*self.num_states)
        s=0
        s_vec.append(s)
        a=np.argmax(self.q_fun_s(s,w))
        a_vec.append(a)
        for t in range(1,100):
            s_next,r_next=self._step(s,a)
            r_vec.append(r_next)
            s_vec.append(s_next)
            a_next=np.argmax(self.q_fun_s(s_next,w))
            a_vec.append(a_next)
            tau=t-n
            if tau>=0:
                delta=np.sum([r_vec[i]-rbar for i in range(tau-1,tau+n-1)])+self.q_fun(s_vec[tau+n],a_vec[tau+n],w)-self.q_fun(s_vec[tau],a_vec[tau],w)
            rbar=rbar+beta*delta
            w=w+alpha*delta*self.q_grad(s,a,w)
            s=s_next
            a=a_next
        return w

class ApproxContinousMDP(ModelFreeEpisodicMDP,ApproxMDP):
    def __init__(self, pmatrix, rewards, num_terminal_states):
        super().__init__(pmatrix, rewards, num_terminal_states)


    def MCPGControl_actorcritic_eligibility(self,alpha,gamma,l):
        theta=np.random.rand(np.sum(np.sum(self.pmatrix,axis=(0,1))>0))
        w=np.random.rand(self.num_states)
        rbar=0
        zt=np.zeros_like(theta)
        zw=np.zeros_like(w)
        for _ in range(1000):
            s=np.random.randint(self.num_states-self.num_terminal_states)
            a=np.random.choice(self.num_actions,p=self.pi_fun_s(s,theta))
            s_next,r_next=self._step(s,a)
            delta=r_next+gamma*self.v_fun(s_next,w)-self.v_fun(s,w)
            zw=l*zw+self.v_grad(s,w)
            zt=l*zt+self.pi_grad(s,a,theta)
            w=w+alpha*delta*zw
            theta=theta+alpha*delta*zt
            s=s_next
        return w,theta




class ApproxEpisodicMDP(ModelFreeEpisodicMDP,ApproxMDP):
    def __init__(self, pmatrix, rewards, num_terminal_states):
        super().__init__(pmatrix, rewards, num_terminal_states)

    def _generate_episode_approx(self,theta):
        s=[]
        a=[]
        r=[]
        s.append(np.random.randint(self.num_states-self.num_terminal_states))
        a.append(np.random.choice(self.num_actions,p=self.pi_fun_s(s[0],theta)))
        flag=1
        t=0
        while flag:
            s_temp,r_temp=self._choice_bivariate(self.pmatrix[:,:,s[t],a[t]])
            t=t+1
            s.append(s_temp)
            r.append(r_temp)
            a_temp=np.random.choice(self.num_actions,p=self.pi_fun_s(s[t],theta))
            a.append(a_temp)
            flag=self.terminal_states.count(s[t])==0
        return np.array(s),np.array(r),np.array(a)


    def gradient_monte_carlo(self,policy,alpha,num_ep):
        for _ in range(num_ep):
            s,r,a=self._generate_episode(policy)
            w=np.zeros(self.num_states)
            for t in range(len(s)-1):
                g=np.sum(s[t:])
                w=w+alpha*(g-self.v_fun(s[t],w))*self.v_grad(s[t],w)
        return w
    
    def semigradient_td0(self,policy,gamma,alpha,num_ep):
        w=np.zeros(self.num_states)
        for _ in range(num_ep):
            s,r,a=self._generate_episode(policy)
            for t in range(len(s)-2):
                w=w+alpha*(r[t]+gamma*self.v_fun(s[t+1],w)-self.v_fun(s[t],w))*self.v_grad(s[t],w)
        return w

    def semigradient_tdn(self, policy, gamma,alpha,n,n_ep):
        w=np.zeros(self.num_states)
        for _ in range(n_ep):
            s,r,a=self._generate_episode(policy)
            for t in range(len(s)-1):
                tau=t-n+1
                if tau>=0:
                    G=np.sum(np.array([gamma**(i-tau-1)*r[i-1] for i in range(tau+1,tau+n+1)]))
                    if tau+n<len(s):
                        G=G+gamma**(n)*self.v_fun(s[tau+n],w)
                    w=w+alpha*(G-self.v_fun(s[tau],w))*self.v_grad(s[tau],w)
        return w
    
    def sarsa(self,alpha,gamma,n_ep):
        w=np.zeros(self.num_states*self.num_actions)
        for i in range(n_ep):
            policy=self.random_policy()
            s,r,a=self._generate_episode(policy)
            for t in range(len(s)-1):
                if t+1==len(s)-1:
                    w=w+alpha*(r[t]-self.q_fun(s[t],a[t],w))*self.q_grad(s[t],a[t],w)
                    continue
                aprime=np.argmax(self.q_fun_s(s[t],w))
                w=w+alpha*(r[t]+gamma*self.q_fun(s[t+1],aprime,w)-self.q_fun(s[t],a[t],w))*self.q_grad(s[t],a[t],w)
        return w

    def sarsa_n(self, gamma,alpha,n,n_ep):
        w=np.random.rand(self.num_states*self.num_actions)
        for _ in range(n_ep):
            a_vec=[]
            r_vec=[]
            s_vec=[]
            t=0
            s=np.random.randint(self.num_states-self.num_terminal_states)
            s_vec.append(s)
            a=np.argmax(self.q_fun_s(s,w))
            a_vec.append(a)
            flag=True
            while flag:
                s_next,r_next=self._step(s,a)
                if self.terminal_states.count(s_next)>0:
                    flag=False
                    continue
                r_vec.append(r_next)
                a_next=np.argmax(self.q_fun_s(s_next,w))
                a_vec.append(a_next)
                tau=t-n
                if tau>=0:
                    G=np.sum(np.array([gamma**(i-tau-1)*r[i-1] for i in range(tau+1,tau+n+1)]))
                    G=G+gamma**(n)*self.q_fun[s_vec[tau+n],a_vec[tau+n]]
                    w=w+alpha*(G-self.q_fun(s_vec[tau],a_vec[tau]))
                s=s_next
                a=a_next
        return w
    
    def semigradient_tdlambda(self,policy,alpha,gamma,l,n_ep):
        w=np.random.rand(self.num_states)
        for _ in range(n_ep):
            s,r,a=self._generate_episode(policy)
            z=0
            for t in range(len(s)-1):
                z=gamma*l*z+self.v_grad(s[t],w)
                delta=r[t]+gamma*self.v_grad(s[t+1],w)-self.v_grad(s[t],w)
                w=w+alpha*delta*z
        return w
    
    def online_tdlambda(self,policy,alpha,gamma,l,n_ep):
        w=np.random.rand(self.num_states)
        for _ in range(n_ep):
            s,r,a=self._generate_episode(policy)
            z=np.zeros_like(self.v_grad(s[0],w))
            vold=0
            for t in range(len(s)-1):
                v=self.v_fun(s[t],w)
                vprime=self.v_fun(s[t+1],w)
                delta=r[t]+gamma*vprime-v
                z=gamma*l*z+(1-alpha*gamma*l*np.dot(z,self.v_grad(s[t],w)))
                w=w+alpha*(delta+v-vold)*z-alpha*(v-vold)*self.v_grad(s[t],w)
                vold=vprime
        return w
    
    def sarsa_lambda(self,policy,alpha,gamma,l,n_ep):
        w=np.random.rand(self.num_states*self.num_actions)
        for _ in range(n_ep):
            s=np.random.randint(self.num_states-self.num_terminal_states)
            a=np.argmax(self.random_q_function()[s])
            z=np.zeros_like(self.q_grad(s,a,w))
            qold=0
            flag=True
            while flag:
                a=np.argmax(self.q_fun_s(s,w))
                s_next,r_next=self._step(s,a)
                if self.terminal_states.count(s_next)>0:
                    flag=False
                    continue
                a_next=np.argmax(self.q_fun_s(s_next,w))
                q=self.q_fun(s,a,w)
                qprime=self.q_fun(s_next,a_next,w)
                delta=r_next+gamma*qprime-q
                z=gamma*l*z+(1-alpha*gamma*l*np.dot(z,self.q_grad(s,a,w)))
                w=w+alpha*(delta+q-qold)*z-alpha*(q-qold)*self.q_grad(s,a,w)
                qold=qprime
                a=a_next
                s=s_next
        return w

    def MCPGControl(self,alpha,gamma,n_ep):
        theta=np.random.rand(np.sum(np.sum(self.pmatrix,axis=(0,1))>0))
        for _ in range(n_ep):
            s,r,a=self._generate_episode_approx(theta)
            for t in range(len(s)-1):
                G=np.sum((gamma**(np.arange(t,len(s)-1)-t))*r[t:])
                theta=theta+alpha*(gamma**t)*G*self.pi_grad_log(s[t],a[t],theta)
        return theta
    
    def MCPGControl_baseline(self,alpha,gamma,n_ep):
        theta=np.random.rand(np.sum(np.sum(self.pmatrix,axis=(0,1))>0))
        w=np.random.rand(self.num_states)
        for _ in range(n_ep):
            s,r,a=self._generate_episode_approx(theta)
            for t in range(len(s)-1):
                G=np.sum((gamma**(np.arange(t,len(s)-1)-t))*r[t:])
                delta=G-self.v_fun(s[t],w)
                w=w+alpha*delta*self.v_grad(s[t],w)
                theta=theta+alpha*(gamma**t)*G*self.pi_grad_log(s[t],a[t],theta)
        return w,theta

    def MCPGControl_actorcritic(self,alpha,gamma,n_ep):
        theta=np.random.rand(np.sum(np.sum(self.pmatrix,axis=(0,1))>0))
        w=np.random.rand(self.num_states)
        for _ in range(n_ep):
            s=np.random.randint(self.num_states-self.num_terminal_states)
            I=1
            flag=True
            while flag:
                a=np.random.choice(self.num_actions,p=self.pi_fun_s(s,theta))
                s_next,r_next=self._step(s,a)
                if self.terminal_states.count(s_next)>0:
                    flag=False
                delta=r_next+gamma*self.v_fun(s_next,w)-self.v_fun(s,w)
                w=w+alpha*delta*self.v_grad(s,w)
                theta=theta+alpha*I*delta*self.pi_grad_log(s,a,theta)
                I=gamma*I
                s=s_next
            
        return w,theta
    
    def MCPGControl_actorcritic_eligibility(self,alpha,gamma,l,n_ep):
        theta=np.random.rand(np.sum(np.sum(self.pmatrix,axis=(0,1))>0))
        w=np.random.rand(self.num_states)
        for _ in range(n_ep):
            s=np.random.randint(self.num_states-self.num_terminal_states)
            zt=np.zeros_like(theta)
            zw=np.zeros_like(w)
            I=1
            flag=True
            while flag:
                a=np.random.choice(self.num_actions,p=self.pi_fun_s(s,theta))
                s_next,r_next=self._step(s,a)
                if self.terminal_states.count(s_next)>0:
                    flag=False
                delta=r_next+gamma*self.v_fun(s_next,w)-self.v_fun(s,w)
                zw=gamma*l*zw+self.v_grad(s,w)
                zt=gamma*l*zt+I*self.pi_grad(s,a,theta)
                w=w+alpha*delta*zw
                theta=theta+alpha*I*delta*zt
                I=gamma*I
                s=s_next
        return w,theta




class RBFEpisodicMDP(ApproxEpisodicMDP):
    def __init__(self, pmatrix, rewards, num_terminal_states):
        super().__init__(pmatrix, rewards, num_terminal_states)

        self.forward_indices_1d=[]
        self.forward_indices_2d=[[] for s in range(self.num_states)]
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if np.sum(self.pmatrix[:,:,s,a])>0:
                    self.forward_indices_2d[s].append(self._encode(s,a))
                    self.forward_indices_1d.append(self._encode(s,a))
        self.forward_indices_1d=np.array(self.forward_indices_1d)
        for s in range(self.num_states):
            self.forward_indices_2d[s]=np.array(self.forward_indices_2d[s])


    def v_fun(self,x,w):
        return np.dot(w,((x-np.arange(self.num_states))/self.num_states)**2)       
    def v_grad(self,x,w):
        return ((x-np.arange(self.num_states))/self.num_states)**2

    def q_fun(self,s,a,w):
        x=self._encode(s,a)
        return np.dot(w,((x-np.arange(self.num_states*self.num_actions))/(self.num_states*self.num_actions))**2)       
    
    def q_grad(self,s,a,w):
        x=self._encode(s,a)
        return ((x-np.arange(self.num_states*self.num_actions))/(self.num_states*self.num_actions))**2

    def pi_fun(self,s,a,theta):
        if np.sum(np.isin(self.forward_indices_1d,self._encode(s,a),True))>0:
            theta_local=theta[np.sum(self.forward_indices_1d==(self.forward_indices_2d[s][:,None]),axis=0).astype(bool)]
            tmp_list=np.array(self.forward_indices_2d[s])
            return theta_local[np.where(tmp_list==self._encode(s,a))[0].item()]**2/np.sum(theta_local**2)
        else:
            return 0
    
    def pi_grad(self,s,a,theta):
        grad=np.zeros_like(theta)
        if np.sum(np.isin(self.forward_indices_1d,self._encode(s,a),True))>0:
            theta_local=theta[np.sum(self.forward_indices_1d==(self.forward_indices_2d[s][:,None]),axis=0).astype(bool)]
            tmp_list=np.array(self.forward_indices_2d[s])
            index_1=np.where(self.forward_indices_1d==self._encode(s,a))[0].item()
            index_2=np.where(tmp_list==self._encode(s,a))[0].item()
            for i in tmp_list:
                if i!=index_2:
                    tmp_1=np.where(self.forward_indices_1d==i)[0].item()
                    tmp_2=np.where(tmp_list==i)[0].item()
                    grad[tmp_1]=-2*theta_local[index_2]**2*theta_local[tmp_2]/np.sum(theta_local**2)**2
            grad[index_1]=2*theta_local[index_2]*np.sum(np.delete(theta_local,index_2)**2)/np.sum(theta_local**2)**2
        return grad
    




    def lstd(self,policy,alpha,epsilon,gamma,n_ep):
        policy=self.random_policy()
        w=np.zeros(self.num_states)
        A=np.eye(self.num_states)/epsilon
        b=0
        for _ in range(n_ep):
            s,r,a=self._generate_episode(policy)
            for t in range(len(s)-2):
                x=self.v_grad(s[t],w)
                xprime=self.v_grad(s[t+1],w)
                v=A.T@(x-gamma*xprime)
                A=A-np.outer((A@x),v)/(1+np.dot(v.T,x))
                b=b+r[t]*x
                w=A@b
        return w 


        
class RBFContinousMDP(ApproxContinousMDP):
    def __init__(self, pmatrix, rewards, num_terminal_states):
        super().__init__(pmatrix, rewards, num_terminal_states)

        self.forward_indices_1d=[]
        self.forward_indices_2d=[[] for s in range(self.num_states)]
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if np.sum(self.pmatrix[:,:,s,a])>0:
                    self.forward_indices_2d[s].append(self._encode(s,a))
                    self.forward_indices_1d.append(self._encode(s,a))
        self.forward_indices_1d=np.array(self.forward_indices_1d)
        for s in range(self.num_states):
            self.forward_indices_2d[s]=np.array(self.forward_indices_2d[s])


    def v_fun(self,x,w):
        return np.dot(w,((x-np.arange(self.num_states))/self.num_states)**2)       
    def v_grad(self,x,w):
        return ((x-np.arange(self.num_states))/self.num_states)**2

    def q_fun(self,s,a,w):
        x=self._encode(s,a)
        return np.dot(w,((x-np.arange(self.num_states*self.num_actions))/(self.num_states*self.num_actions))**2)       
    
    def q_grad(self,s,a,w):
        x=self._encode(s,a)
        return ((x-np.arange(self.num_states*self.num_actions))/(self.num_states*self.num_actions))**2

    def pi_fun(self,s,a,theta):
        if np.sum(np.isin(self.forward_indices_1d,self._encode(s,a),True))>0:
            theta_local=theta[np.sum(self.forward_indices_1d==(self.forward_indices_2d[s][:,None]),axis=0).astype(bool)]
            tmp_list=np.array(self.forward_indices_2d[s])
            return theta_local[np.where(tmp_list==self._encode(s,a))[0].item()]**2/np.sum(theta_local**2)
        else:
            return 0
    
    def pi_grad(self,s,a,theta):
        grad=np.zeros_like(theta)
        if np.sum(np.isin(self.forward_indices_1d,self._encode(s,a),True))>0:
            theta_local=theta[np.sum(self.forward_indices_1d==(self.forward_indices_2d[s][:,None]),axis=0).astype(bool)]
            tmp_list=np.array(self.forward_indices_2d[s])
            index_1=np.where(self.forward_indices_1d==self._encode(s,a))[0].item()
            index_2=np.where(tmp_list==self._encode(s,a))[0].item()
            for i in tmp_list:
                if i!=index_2:
                    tmp_1=np.where(self.forward_indices_1d==i)[0].item()
                    tmp_2=np.where(tmp_list==i)[0].item()
                    grad[tmp_1]=-2*theta_local[index_2]**2*theta_local[tmp_2]/np.sum(theta_local**2)**2
            grad[index_1]=2*theta_local[index_2]*np.sum(np.delete(theta_local,index_2)**2)/np.sum(theta_local**2)**2
        return grad
    






'''
r,p=continous_example()
np.random.seed(0)
mdp=knownMDP(p,r)


policy=mdp.random_policy()

v=mdp.evaluate_policy(policy,1,0.00001)

pi=mdp.optimal_policy_value_iteration(0.8,0.00001)
'''
def episodic_example():
    ### Student MDP (silver),
    #States:
    #0)Class 1
    #1)Class 2
    #2)Class 3
    #3)Facebook
    #4)Pub
    #5)Sleep

    #Actions
    #0)Study
    #1)Go to Facebook
    #2)Quit facebook
    #3)Go to Pub
    #4)Quit Pub
    #5)Sleep

    #Rewards)
    #0)-2
    #1)-1
    #2)0
    #3)1
    #4)+10

    p=np.zeros((6,5,6,6))
    p[3,1,3,1]=1 
    p[0,2,3,2]=1
    p[3,1,0,1]=1
    p[1,0,0,0]=1
    p[2,0,1,0]=1
    p[5,2,1,5]=1   
    p[5,4,2,0]=1    
    p[4,3,2,3]=1
    p[0,2,4,4]=0.2
    p[1,2,4,4]=0.4
    p[2,2,4,4]=0.4
    p[5,2,5,:]=1
    r=np.array([-2,-1,0,1,10])
    return r,p

'''
r,p=episodic_example()
epmdp=ModelFreeEpisodicMDP(p,r,1)
policy=epmdp.random_policy()
v=epmdp.evaluate_policy_mcmc(policy,0.8,100)
policy=epmdp.random_policy()
q_est=epmdp.evaluate_q_off_mcmc(policy,0.8,100)
policy=epmdp.random_policy()
mc_opt_policy=epmdp.onpolicy_mcmc_optimal(0.10,0.8,100)
mc_opt_policy=epmdp.offpolicy_mcmc_optimal(0.8,100)
v=epmdp.evaluate_policy_td0(policy,0.8,0.01,100)
v=epmdp.evaluate_policy_tdn(policy,0.8,0.01,1,100)
q=epmdp.sarsa_optimal_q(0.8,0.01,100)
q=epmdp.sarsa_optimal_q_n(0.8,0.01,1,100)
q=epmdp.tree_optimal_q_n(0.8,0.01,1,100)
b=epmdp.random_policy()
q=epmdp.off_sarsa_optimal_q_n(0.8,0.01,1,b,100)
q=epmdp.sigma_optimal_q_n(0.8,0.01,1,b,100)
policy=epmdp.q_learning(0.8,0.01,100)
policy=epmdp.double_q_learning(0.8,0.01,100)



'''



'''
r,p=continous_example()
mdpmodel=ModelMDP(p,r)
q=mdpmodel.one_step_tab_q_planning(0.01,0.8)
q,model=mdpmodel.dyna_q(0.01,0.8)
q,model=mdpmodel.prioritized_sweeping(0.01,0.8)
'''

'''
r,p=episodic_example()
rbfmdp=RBFEpisodicMDP(p,r,1)
policy=rbfmdp.random_policy()
w=rbfmdp.gradient_monte_carlo(policy,0.01,100)
w=rbfmdp.semigradient_td0(policy,0.8,0.01,100)
w=rbfmdp.semigradient_tdn(policy,0.8,0.01,1,100)
w=rbfmdp.lstd(policy,0.01,0.1,0.8,100)
w=rbfmdp.sarsa(0.01,0.8,100)
w=rbfmdp.sarsa_n(0.8,0.01,1,100)
'''

'''
r,p=continous_example()
rbfmdp=RBFEpisodicMDP(p,r,1)
w=rbfmdp.semigradientsarsa(1,1)
w=rbfmdp.semigradientsarsa_n(1,1,1)
'''
'''
np.random.seed(0)
r,p=episodic_example()
rbfmdp=RBFEpisodicMDP(p,r,1)
policy=rbfmdp.random_policy()
w=rbfmdp.semigradient_tdlambda(policy,1,0.8,1,10)
w=rbfmdp.online_tdlambda(policy,0.1,0.8,1,10)
w=rbfmdp.sarsa_lambda(policy,0.01,0.8,0.1,10)
theta=rbfmdp.MCPGControl(0.01,0.8,10)
w,theta=rbfmdp.MCPGControl_baseline(0.01,0.8,10)
w,theta=rbfmdp.MCPGControl_actorcritic(0.01,0.8,10)
w,theta=rbfmdp.MCPGControl_actorcritic_eligibility(0.01,0.8,1,10)
'''
np.random.seed(0)
r,p=continous_example()
rbfmdp=RBFContinousMDP(p,r,1)
w,theta=rbfmdp.MCPGControl_actorcritic_eligibility(0.1,0.8,1)
