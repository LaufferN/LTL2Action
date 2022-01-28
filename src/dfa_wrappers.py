"""
This is a simple wrapper that will include DFA goals to any given environment.
It progress the state of the DFA as the agent interacts with the envirionment.

However, each environment must implement the followng functions:
    - *get_events(...)*:       Returns the propositions that currently hold on
                               the environment.
    - *get_propositions(...)*: Maps the objects in the environment to a set of
                               propositions that can be referred to.

Notes about DFAEnv:
    - The episode ends if the DFA goal is progressed to True or False.
    - If the DFA goal becomes True, then an extra +1 reward is given to the agent.
    - If the DFA goal becomes False, then an extra -1 reward is given to the agent.
    - Otherwise, the agent gets the same reward given by the original environment.
"""


import numpy as np
import gym
from gym import spaces
from copy import deepcopy
import dfa_progression, random
from dfa_samplers import getDFASampler, draw
import networkx as nx

class DFAEnv(gym.Wrapper):
    def __init__(self, env, progression_mode="full", dfa_sampler=None, intrinsic=0.0):
        """
        DFA environment
        --------------------
        It adds a DFA objective to the current environment
            - The observations become a dictionary with an added "text" field
              specifying the DFA objective
            - It also automatically progress the DFA and generates an
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas (Check this later)
        progression_mode:
            - "full": the agent gets the full, progressed DFA as part of the observation
            - "partial": the agent sees which propositions (individually) will progress or falsify the formula
            - "none": the agent gets the full, original DFA as part of the observation
        """
        super().__init__(env)
        self.progression_mode = progression_mode
        self.propositions = self.env.get_propositions()
        self.sampler = getDFASampler(dfa_sampler, self.propositions)

        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.known_progressions = {}
        self.intrinsic = intrinsic


    def sample_dfa_goal(self):
        # This function must return a DFA for a task.
        # Format: networkx graph
        # NOTE: The propositions must be represented by a char
        dfa = self.sampler.sample()
        return dfa

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        raise NotImplementedError

    def reset(self):
        self.known_progressions = {}
        self.obs = self.env.reset()

        # Defining a DFA goal
        self.dfa_goal     = self.sample_dfa_goal()
        self.dfa_original = deepcopy(self.dfa_goal)

        # Adding the DFA goal to the observation
        if self.progression_mode == "partial":
            dfa_obs = {'features': self.obs,'progress_info': self.progress_info(self.dfa_goal)}
        else:
            dfa_obs = {'features': self.obs,'text': self.dfa_goal}
        return dfa_obs


    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the DFA
        truth_assignment = self.get_events(self.obs, action, next_obs)
        #print("Before:")
        #for n in self.dfa_goal.nodes:
        #    print(n, self.dfa_goal.nodes[n])
        #for e in self.dfa_goal.edges:
        #    print(e, self.dfa_goal.edges[e])
        #print("truth_assignment", truth_assignment)
        #input()
        dfa_reward, _, dfa_done = self.progression(self.dfa_goal, truth_assignment)
        #print("After:")
        #for n in self.dfa_goal.nodes:
        #    print(n, self.dfa_goal.nodes[n])
        #for e in self.dfa_goal.edges:
        #    print(e, self.dfa_goal.edges[e])
        #print("truth_assignment", truth_assignment)
        #input()
        self.obs = next_obs

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            dfa_obs = {'features': self.obs,'text': self.dfa_goal}
        elif self.progression_mode == "none":
            dfa_obs = {'features': self.obs,'text': self.dfa_original}
        elif self.progression_mode == "partial":
            dfa_obs = {'features': self.obs, 'progress_info': self.progress_info(self.dfa_goal)}
        else:
            raise NotImplementedError

        reward  = original_reward + dfa_reward
        done    = env_done or dfa_done
        return dfa_obs, reward, done, info

    def progression(self, dfa, truth_assignment):
        propositions = self.env.get_propositions()
        return dfa_progression.progress_and_clean(dfa, truth_assignment, propositions)

    # # X is a vector where index i is 1 if prop i progresses the formula, -1 if it falsifies it, 0 otherwise.
    def progress_info(self, dfa):
        propositions = self.env.get_propositions()
        progression_info = np.zeros(len(self.propositions))
        for i in range(len(propositions)):
            _, progression_info[i], _ = self.progression(dfa, propositions[i])
        return progression_info

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()


class NoDFAWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Removes the DFA from a DFAEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space = env.observation_space
        # self.observation_space =  env.observation_space['features']

    def reset(self):
        obs = self.env.reset()
        # obs = obs['features']
        # obs = {'features': obs}
        return obs

    def step(self, action):
        # executing the action in the environment
        obs, reward, done, info = self.env.step(action)
        # obs = obs['features']
        # obs = {'features': obs}
        return obs, reward, done, info

    def get_propositions(self):
        return list([])

def draw(G, path):
    A = nx.drawing.nx_agraph.to_agraph(G)
    A.layout('dot')
    A.draw(path)

if __name__ == '__main__':
    props = "abcdefghijkl"
    env = gym.make("Letter-7x7-v3")
    env.seed(1)
    dfaEnv = DFAEnv(env, "full", "Default", 0)
    dfaEnv.reset()
    dfa = dfaEnv.dfa_goal
    #draw(dfa, "sample_dfa.png")
    print("-------------------")
    print(dfaEnv.progression(dfa, "a"))
    draw(dfa, "sample_dfa.png")
    input()
    print(dfaEnv.progression(dfa, "j"))
    draw(dfa, "sample_dfa.png")
    input()
    print(dfaEnv.progression(dfa, "d"))
    draw(dfa, "sample_dfa.png")
    input()
    print(dfaEnv.progression(dfa, "c"))
    draw(dfa, "sample_dfa.png")
    input()
    for e in dfa.nodes:
        print(e, dfa.nodes[e])
