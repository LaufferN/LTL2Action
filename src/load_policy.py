import utils
from envs.gridworld.gridworld_env import GridworldEnv
from dfa_samplers import draw

"""
- get dfa viz from DISS repo
- reweight dfas based on a random walk accepting probability, and make plot of (dfa size x avg acceptence)
"""

class GridworldPlanner:

    def __init__(self):

        env_type = 'Gridworld-v3'
        model_dir = 'storage/RGCN_8x32_ROOT_SHARED_Universal_0.05_Gridworld-v3_seed:1_epochs:4_bs:256_fpp:None_dsc:0.95_lr:0.0003_ent:0.01_clip:0.2_prog:full_use-dfa:True_use_mean_guard_embed:False_use_onehot_guard_embed:False_give_mdp_state_to_gnn:False'
        # model_dir = 'storage/RGCN_8x32_ROOT_SHARED_FixedGridworld_Gridworld-v2_seed:1_epochs:4_bs:256_fpp:None_dsc:0.95_lr:0.0003_ent:0.01_clip:0.2_prog:full_use-dfa:True_use_mean_guard_embed:False_use_onehot_guard_embed:False_give_mdp_state_to_gnn:False'
        ignoreLTL = False
        gnn = 'RGCN_8x32_ROOT_SHARED'
        progression_mode = 'full'
        recurrence = 1
        dumb_ac = False
        use_dfa = True
        use_mean_guard_embed=False
        env_seed = None
        ltl_sampler = 'Universal_0.05'
        # ltl_sampler = 'FixedGridworld'

        self.env = utils.make_env(env_type, progression_mode, ltl_sampler, env_seed, 0, False, use_dfa=use_dfa)

        self.agent = utils.Agent(self.env, self.env.observation_space, self.env.action_space, model_dir + "/train", 
            ignoreLTL, progression_mode, gnn, recurrence=recurrence, dumb_ac=dumb_ac, use_dfa=use_dfa, use_mean_guard_embed=use_mean_guard_embed)

if __name__ == '__main__':
    gw_planner = GridworldPlanner()
    obs = gw_planner.env.reset()
    draw(obs['text'], 'sample_dfa.png')
    # print(obs['features'].transpose())
    print(gw_planner.env.gw.to_string(gw_planner.env.state))
    while True:
        action = gw_planner.agent.get_action(obs)
        print(gw_planner.env.actions[action])
        obs, reward, done, _ = gw_planner.env.step(action)
        # print(obs['features'].transpose())
        print(gw_planner.env.gw.to_string(gw_planner.env.state))
        if done:
            if reward > 0:
                print('success!')
            else:
                print('failure')
            break
        input()
