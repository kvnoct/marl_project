import gym
from a2c import A2C
import torch
import os 
from torch.utils.tensorboard import SummaryWriter
import argparse 
from wrappers import TimeLimit

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type = int)
    parser.add_argument('--model-save-path', type = str)
    parser.add_argument('--log-path', type = str)
    
    #Env arguments
    parser.add_argument('--map-size', type = str, choices = ['tiny', 'small', 'medium', 'large'], default='tiny')
    parser.add_argument('--num-agent', type = int, default = 2)
    parser.add_argument('--difficulty', type = str, choices = ['easy', 'normal', 'hard'], default = 'normal')

    #Algo arguments
    parser.add_argument('--use-gae',  default=False, action = "store_true")
    parser.add_argument('--lr', type = float, default = 3e-4)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--gae-lambda', type = float, default = 0.95)
    parser.add_argument('--adam-eps', type = float, default = 1e-3)
    parser.add_argument('--total-timesteps', type = int, default = 1000000)
    parser.add_argument('--n-steps', type = int, default = 5)

    args = parser.parse_args()
    return args

def make_env(args):
    if args.difficulty == "normal":
        difficulty_str = ""
    else:
        difficulty_str = "-" + args.difficulty

    env = gym.make(f"rware:rware-{args.map_size}-{args.num_agent}ag{difficulty_str}-v1")

    # env = TimeLimit(env, args.epilen)
    return env

def main(args = get_args()):
    torch.manual_seed(int(args.seed))

    if os.path.isdir(f"{args.log_path}") == False:
        os.mkdir(f"{args.log_path}")

    if os.path.isdir(f"{args.model_save_path}") == False:
        os.mkdir(f"{args.model_save_path}")

    env = make_env(args)
    writer = SummaryWriter(f"{args.log_path}/run_{args.seed}")

    obs_space = env.observation_space
    action_space = env.action_space

    agents = [A2C(i, obs_space, action_space,
                  num_steps = args.n_steps,
                  lr = args.lr,
                  gamma = args.gamma,
                  use_gae = args.use_gae,
                  gae_lambda = args.gae_lambda,
                  adam_eps = args.adam_eps) for i in range(env.n_agents)]

    #Episode is finished
    obs = env.reset()
    rewards = [0 for _ in range(len(agents))]
    
    #Get the initial obs
    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(torch.tensor(obs[i]))
        agents[i].storage.to('cpu')
    done = [False for _ in range(len(agents))]

    total_updates = int(args.total_timesteps / args.n_steps)
    
    t = 0
    for j in range(total_updates):
        #N-steps rollout
        env.render()
        for _ in range(args.n_steps):
            t += 1

            #Get the action
            with torch.no_grad():
                n_value, n_action, n_log_probs = zip( *[agent.model.act(agent.storage.obs[0]) for agent in agents] )
                n_action = [action.item() for action in n_action]

            #Step on env
            obs, reward, done, infos = env.step(n_action)

            rewards = [rewards[rew_idx] + reward[rew_idx] for rew_idx in range(len(rewards))]

            #Get the mask
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done])

            #Copy the state transition to agent's on-policy storage
            for i in range(len(agents)):
                agents[i].storage.insert(torch.tensor(obs[i]),
                                        torch.tensor(n_action[i]),
                                        torch.tensor(n_log_probs[i]), 
                                        n_value[i],
                                        torch.tensor(reward[i]),
                                        masks[i])

        for agent in agents:
            agent.compute_returns()

        for agent in agents:
            loss = agent.update([a.storage for a in agents])
            for k, v in loss.items():
                writer.add_scalar(f"agent{agent.agent_id}/{k}", v, t)

        for agent in agents:
            agent.storage.after_update()
        
        #Record cumulative rewards
        for agent_idx in range(len(agents)):
            writer.add_scalar(f"agent{agents[agent_idx].agent_id}/cumulative_rewards", rewards[agent_idx], t)
            writer.add_scalar(f"agent{agents[agent_idx].agent_id}/smoothed_rewards", rewards[agent_idx]/t, t)

    

    #save model
    for agent in agents:
        torch.save(agent.model, f"{args.model_save_path}/model_agent{agent.agent_id}.pt")

    env.close()

if __name__ == "__main__":
    main()