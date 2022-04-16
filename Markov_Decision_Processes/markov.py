import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from hiivemdptoolbox.hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
from hiivemdptoolbox.hiive.mdptoolbox.example import forest, openai

def easy_mdp():
    T, R = openai("FrozenLake-v1")
    
    return T, R

def hard_mdp():
    T, R = forest(S=1000, r1=1000, r2=10000, p=.005)
    
    return T, R
            
def value(T, R, discount, max_iter, verbose=False):
    model = ValueIteration(T, R, discount, max_iter=max_iter)
    model.run()
    rewards = []
    errors = []
    
    if verbose:
        for dic in model.run_stats:
            rewards.append(dic["Reward"])
            errors.append(dic["Error"])

    return model.iter, model.policy, rewards, errors

def policy(T, R, discount, max_iter, verbose=False):
    model = PolicyIteration(T, R, discount, max_iter=max_iter)
    model.run()
    rewards = []
    errors = []
    
    if verbose:
        for dic in model.run_stats:
            rewards.append(dic["Reward"])
            errors.append(dic["Error"])
    
    return model.iter, model.policy, rewards, errors
 		
def q_learning(T, R, discount, max_iter, verbose=False):    
    model = QLearning(T, R, discount, alpha=.2, alpha_decay=1, epsilon_decay=0.95)
    total_rewards = []
    total_errors = []
    out = []
    
    for i in range(max_iter):
        if verbose and i % 5 == 0:
            rewards = []
            errors = []
        
        model.run()
        
        if verbose and i % 5 == 0:
            for dic in model.run_stats:
                rewards.append(dic["Reward"])
                errors.append(dic["Error"])
            total_rewards.append(sum(rewards)/max_iter)
            total_errors.append(sum(errors)/max_iter)
            out.append(i)
    
    return model.policy, total_rewards, total_errors, out

def graph_rewards(value_rewards, pol_rewards, q_rewards, out, type):
    plt.plot(value_rewards, label="Value Iteration")
    plt.plot(pol_rewards, label="Policy Iteration")
    title = f"Reward over iteration for {type} Problems"
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend()
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf()    
    
    plt.plot(out, q_rewards, label="QLearning")
    title = f"Rewards for QLearner for {type} Problems"
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend()
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf() 
    
def graph_errors(value_errs, pol_errs, q_errs, out, type):
    plt.plot(value_errs, label="Value Iteration")
    plt.plot(pol_errs, label="Policy Iteration")
    title = f"Error over iteration for {type} Problems"
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf()    
    
    plt.plot(out, q_errs, label="QLearning")
    title = f"Error for QLearner for {type} Problems"
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf() 

def graph_times(value_times, pol_times, q_times, discounts, type):
    plt.plot(discounts, value_times, label="Value Iteration")
    plt.plot(discounts, pol_times, label="Policy Iteration")
    plt.plot(discounts, q_times, label="QLearning")
    title = f"Discount vs Time for {type} Problems"
    plt.title(title)
    plt.xlabel("Discount Rate")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf()
    
def graph_iters(value_iters, pol_iters, discounts, type):
    plt.plot(discounts, value_iters, label="Value Iteration")
    plt.plot(discounts, pol_iters, label="Policy Iteration")
    title = f"Discount vs Iterations for {type} Problems"
    plt.title(title)
    plt.xlabel("Discount Rate")
    plt.ylabel("Iterations")
    plt.legend()
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf()
    
if __name__ == "__main__":
    discounts_og = [x for x in range(50, 100, 10)]
    discounts_og.append(99)
    discounts_og.append(99.9)
    arr = np.array(discounts_og)
    arr = arr/100
    discounts = arr.tolist()
    types = ["Easy", "Hard"]
    
    for t in types:
        if t == "Easy":
            T, R = easy_mdp()
            max_iter = 1000
        elif t == "Hard":
            T, R = hard_mdp()
            max_iter = 100
        
        value_times = []
        pol_times = []
        q_times = []
        
        value_iters = []
        pol_iters = []
        
        for discount in discounts:
            curr_time = datetime.now()
            iter, val_pol, _, _ = value(T, R, discount, max_iter)
            finish_time = (datetime.now() - curr_time).total_seconds()
            value_times.append(finish_time)
            value_iters.append(iter)
            
            curr_time = datetime.now()
            iter, pol_pol, _, _ = policy(T, R, discount, max_iter)
            finish_time = (datetime.now() - curr_time).total_seconds()
            pol_times.append(finish_time)
            pol_iters.append(iter)
            
            curr_time = datetime.now()
            q_pol, _, _, _ = q_learning(T, R, discount, max_iter)
            finish_time = (datetime.now() - curr_time).total_seconds()
            q_times.append(finish_time)
            
            if t == "Easy" and val_pol == pol_pol:
                print("Easy Val == Pol:", discount)
            elif t == "Hard" and val_pol != pol_pol:
                print("Hard Val != Pol:", discount)
                
        graph_times(value_times, pol_times, q_times, discounts, t)
        graph_iters(value_iters, pol_iters, discounts, t)
    
        discount = .9
    
        val_iter, val_pol, val_rewards, val_errors = value(T, R, discount, max_iter, verbose=True)
        
        pol_iter, pol_pol, pol_rewards, pol_errors = policy(T, R, discount, max_iter, verbose=True)
        
        q_pol, q_rewards, q_errors, out = q_learning(T, R, discount, max_iter, verbose=True)
        
        print(val_pol)
        print(pol_pol)
        print(q_pol)
        print(val_iter, pol_iter)
        
        graph_rewards(val_rewards, pol_rewards, q_rewards, out, t)
        graph_errors(val_errors, pol_errors, q_errors, out, t)