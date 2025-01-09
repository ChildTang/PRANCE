import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Trick 8: Orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)

        self.initial_std = 0.3
        self.std = self.initial_std

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc_mu, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.fc_mu(x))
        mu = (mu + 1) / 2

        std = torch.ones_like(mu) * self.std
        std = std.to(x.device)

        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class PPO:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        total_train_step,
        token_mode,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
        self.critic = ValueNet(state_dim, hidden_dim)

        # Trick 9
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5
        )

        self.state_dim = state_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.total_train_step = total_train_step
        self.token_mode = token_mode

        self.train_info = None
        self.action_dim = action_dim
        self.on_train_step = 0

        # for debug
        self.critic_log = []
        self.train_action_dict = {
            "1": [],
            "2": [],
            "3": [],
        }
        self.train_return_dict = {
            "total_return": [],
            "acc": [],
            "flops": [],
            "prune": [],
            "merge": [],
        }
        self.train_advantage_dict = {
            "mean": [],
            "max": [],
            "min": [],
        }

    def take_action(self, state, is_training=False):
        mu, sigma = self.actor(state)

        if is_training:
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            action = torch.clamp(action, 0, 1)  # Clamp the action to the range [0, 1]
        else:
            action = mu

        return np.array(action.tolist()), np.array(mu.tolist())

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0

        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()

        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self, transition_dict):
        device = self.actor.fc1.weight.device

        states = torch.stack(transition_dict["states"]).to(device)
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .view(-1, self.action_dim)
            .to(device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(device)
        )
        next_states = torch.stack(transition_dict["next_states"]).to(device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(device)
        )

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        critic_value = self.critic(states)
        td_delta = td_target - critic_value
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            device
        )

        #  data record
        self.critic_log.append(torch.max(critic_value).detach().cpu().numpy().item())
        self.train_advantage_dict["mean"].append(
            torch.mean(advantage).detach().cpu().numpy().item()
        )
        self.train_advantage_dict["max"].append(
            torch.max(advantage).detach().cpu().numpy().item()
        )
        self.train_advantage_dict["min"].append(
            torch.min(advantage).detach().cpu().numpy().item()
        )

        # Trick 1
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # The action follows a normal distribution
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)

            # Trick 5: policy entropy
            dist_entropy = action_dists.entropy().sum(1, keepdim=True)

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # Trick 5: policy entropy
            actor_loss = torch.mean(-torch.min(surr1, surr2) - 0.01 * dist_entropy)
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            # Trick 7
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.on_train_step += 1
        # Update the exploration rate of the actor
        self.update_actor_exploration_rate(self.on_train_step)

    def train_agent(self, env, max_action=False):
        self.actor.train()
        self.critic.train()

        env.reset(mode="train")
        batch_size = env.args.batch_size

        transition_dict = {
            "states": [],
            "actions": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
        }

        token_remain_list = []

        episode_done = False
        data_id = 0
        while not episode_done:
            data_done = False
            sample_id = 0

            data_return = 0
            acc_return = 0
            flops_return = 0
            prune_return = 0
            merge_return = 0

            while not data_done:
                if sample_id == 0:
                    # Note here, otherwise init_config will be changed
                    config = copy.deepcopy(env.init_config)
                    pre_action = np.ones((batch_size, self.action_dim))
                    token_remain = np.ones(batch_size)
                else:
                    pre_action, action_record = self.take_action(
                        state, is_training=True
                    )
                    config = self.action_to_config(
                        pre_action,
                        env.choices,
                        config,
                        block_num=sample_id * 3,
                    )

                    self.train_action_dict[str(sample_id)].append(
                        np.mean(action_record, axis=0)
                    )

                    if self.token_mode == "prune" or self.token_mode == "merge":
                        token_remain *= action_record[:, -1]
                    elif self.token_mode == "prune_merge":
                        token_remain *= action_record[:, -2]
                        token_remain *= action_record[:, -1]

                re_info, pre_reward, data_done, episode_done = env.step(config)
                next_state = torch.mean(re_info["k"], dim=-1)

                # next_state = torch.mean(re_info["q"], dim=-1)
                # next_state = torch.mean(re_info["v"], dim=-1)
                # next_state = re_info["cls_token"]

                # q = torch.mean(re_info["q"], dim=-1)
                # k = torch.mean(re_info["k"], dim=-1)
                # v = torch.mean(re_info["v"], dim=-1)
                # next_state = torch.cat([q, k, v], dim=1)

                reward_list = self.reward_process(
                    pre_reward,
                    pre_action,
                    sample_id,
                    mode="train",
                )

                if sample_id != 0:
                    transition_dict["states"].extend(state)
                    transition_dict["actions"].extend(pre_action)
                    transition_dict["rewards"].extend(reward_list["total_reward"])
                    transition_dict["next_states"].extend(next_state)
                    if data_done:
                        transition_dict["dones"].extend(np.ones(batch_size, dtype=bool))
                    else:
                        transition_dict["dones"].extend(
                            np.zeros(batch_size, dtype=bool)
                        )

                data_return += reward_list["total_reward"] / 4
                acc_return += reward_list["acc"] / 4
                flops_return += reward_list["flops"] / 4

                if self.token_mode == "prune":
                    prune_return += reward_list["prune_rate"] / 4
                elif self.token_mode == "merge":
                    merge_return += reward_list["merge_rate"] / 4
                elif self.token_mode == "prune_merge":
                    prune_return += reward_list["prune_rate"] / 4
                    merge_return += reward_list["merge_rate"] / 4

                state = next_state
                sample_id += 1

            data_id += 1

            self.update(transition_dict)

            self.train_return_dict["total_return"].append(data_return)
            self.train_return_dict["acc"].append(acc_return)
            self.train_return_dict["flops"].append(flops_return)
            if self.token_mode == "prune":
                self.train_return_dict["prune"].append(prune_return)
            elif self.token_mode == "merge":
                self.train_return_dict["merge"].append(merge_return)
            elif self.token_mode == "prune_merge":
                self.train_return_dict["prune"].append(prune_return)
                self.train_return_dict["merge"].append(merge_return)

            token_remain_list.append(token_remain)

            if data_id % 10 == 0:
                re = np.mean(self.train_return_dict["total_return"][-batch_size:])
                pr_token = np.mean(token_remain_list[-batch_size:])
                acc = np.mean(
                    pre_reward["acc"][0].cpu().numpy(),
                )
                target_acc = np.mean(
                    pre_reward["target_acc"][0].cpu().numpy(),
                )
                flops = np.mean(pre_reward["flops"] / 1e8)

                print(
                    "step:{}, re: {:.2f}, acc:{:.2f}, target_acc:{:.2f}, flops:{:.2f},token_remain:{:.2f}, explore:{:.4f}".format(
                        # episode,
                        data_id,
                        re,
                        acc,
                        target_acc,
                        flops,
                        pr_token,
                        self.actor.std,
                    )
                )

            if data_id == 750 or episode_done:
                # if data_id == 10:
                return (
                    self.train_return_dict,
                    self.train_action_dict,
                    self.critic_log,
                    self.train_advantage_dict,
                )

    def test_agent(
        self,
        env,
    ):
        print("*" * 40)
        print("Start testing!!!")

        self.actor.eval()
        self.critic.eval()

        env.reset(mode="test")
        batch_size = env.args.batch_size

        return_list = []
        token_remain_list = []

        pre_reward_list = {
            "acc1": [],
            "flops": [],
            "parameters": [],
            "true_token_remain": [],
        }

        episode_done = False
        data_id = 0
        while not episode_done:
            data_done = False
            sample_id = 0
            last_id = False

            while not data_done:
                if sample_id == 0:
                    # Note here, otherwise init_config will be changed
                    config = copy.deepcopy(env.init_config)
                    pre_action = np.ones((batch_size, self.action_dim))
                    token_remain = np.ones(batch_size)
                else:
                    pre_action, _ = self.take_action(state)
                    config = self.action_to_config(
                        pre_action,
                        env.choices,
                        config,
                        block_num=sample_id * 3,
                    )
                    if self.token_mode == "prune" or self.token_mode == "merge":
                        token_remain *= pre_action[:, -1]
                    elif self.token_mode == "prune_merge":
                        token_remain *= pre_action[:, -2]
                        token_remain *= pre_action[:, -1]

                re_info, pre_reward, data_done, episode_done = env.step(config)
                next_state = torch.mean(re_info["k"], dim=-1)

                # next_state = torch.mean(re_info["q"], dim=-1)
                # next_state = torch.mean(re_info["v"], dim=-1)
                # next_state = re_info["cls_token"]

                # q = torch.mean(re_info["q"], dim=-1)
                # k = torch.mean(re_info["k"], dim=-1)
                # v = torch.mean(re_info["v"], dim=-1)
                # next_state = torch.cat([q, k, v], dim=1)

                state = next_state
                sample_id += 1

                if sample_id == 4:
                    last_id = True

                reward_list = self.reward_process(
                    pre_reward,
                    pre_action,
                    sample_id,
                    mode="test",
                    last_id=last_id,
                )

            data_id += 1

            token_remain_list.append(token_remain)
            return_list.append(reward_list["total_reward"])

            pr_token = np.mean(token_remain_list[-batch_size:])
            re = return_list[-batch_size:]
            acc = pre_reward["acc"][0].cpu().numpy()
            target_acc = pre_reward["target_acc"][0].cpu().numpy()
            flops = pre_reward["flops"] / 1e8
            param = pre_reward["param"] / 1e6

            pre_reward_list["acc1"].extend(acc)
            pre_reward_list["flops"].extend(flops)
            pre_reward_list["parameters"].extend(param)

            if data_id % 10 == 0:
                print(
                    "step:{}, re: {:.2f}, acc:{:.2f}, target_acc:{:.2f}, flops:{:.2f},token_remain:{:.2f}, param:{:.2f}".format(
                        # episode,
                        data_id,
                        np.mean(re),
                        np.mean(acc),
                        np.mean(target_acc),
                        np.mean(flops),
                        pr_token,
                        np.mean(param),
                    )
                )

            # if data_id == 10:
            #     pre_reward_list["true_token_remain"] = token_remain_list
            #     return pre_reward_list

        pre_reward_list["true_token_remain"] = token_remain_list

        return pre_reward_list

    def action_to_config(
        self,
        action,
        choices,
        config,
        block_num,
    ):
        ###############################################################
        # change model config
        # mlp_ratio = np.array([2, 4, 6])
        # embed_dim = np.array([176, 192, 216, 240])
        mlp_ratio = np.array(choices["mlp_ratio"])
        embed_dim = np.array(choices["embed_dim"])

        mlp = np.round(np.clip(action[:, 0:3], 0, 1) * (len(mlp_ratio) - 1)).astype(int)
        embed = np.round(np.clip(action[:, 3:6], 0, 1) * (len(embed_dim) - 1)).astype(
            int
        )
        mlp_choice = mlp_ratio[mlp]
        embed_choice = embed_dim[embed]

        config["mlp_ratio"][:, block_num : block_num + 3] = mlp_choice
        config["embed_dim"][:, block_num : block_num + 3] = embed_choice
        ###############################################################
        # change compression rate
        if self.token_mode == "prune":
            prune = action[:, -1]
            # prune = 0.5 + 0.5 * prune
            config["prune_granularity"][:, block_num] = prune
        elif self.token_mode == "merge":
            merge = action[:, -1]
            config["merge_granularity"][:, block_num] = merge
        elif self.token_mode == "prune_merge":
            prune = action[:, -2]
            merge = action[:, -1]
            config["prune_granularity"][:, block_num] = prune
            config["merge_granularity"][:, block_num] = merge
        ###############################################################

        return config

    def reward_process(
        self,
        raw_reward,
        raw_action,
        sample_id,
        mode,
        last_id=False,
        flops_ratio=0.008,
        prune_ratio=0.1,
        merge_ratio=0.1,
    ):
        re_list = {}
        if mode == "train" or last_id:
            acc = raw_reward["acc"][0].cpu().numpy() / 100
            target_acc = raw_reward["target_acc"][0].cpu().numpy() / 100
            flops = raw_reward["flops"] / 1e8 * flops_ratio

            # Optimize samples that were incorrectly classified by the original ViT
            acc[acc == target_acc] = 1

            if self.token_mode == "prune":
                prune_rate = raw_action[:, -1] * prune_ratio
                reward = acc - flops - prune_rate

                re_list["prune_rate"] = prune_rate

            elif self.token_mode == "merge":
                merge_rate = raw_action[:, -1] * merge_ratio
                reward = acc - flops - merge_rate

                re_list["merge_rate"] = merge_rate

            elif self.token_mode == "prune_merge":
                prune_rate = raw_action[:, -2] * prune_ratio
                merge_rate = raw_action[:, -1] * merge_ratio
                reward = acc - flops - prune_rate - merge_rate

                re_list["prune_rate"] = prune_rate
                re_list["merge_rate"] = merge_rate

            # reward = acc - flops
            # reward = acc
            # reward = -flops

            re_list["total_reward"] = reward
            re_list["acc"] = acc
            re_list["flops"] = flops

        return re_list

    def update_actor_exploration_rate(self, on_train_step):
        std = self.actor.initial_std * (1 - on_train_step / self.total_train_step)
        self.actor.std = max(0.05, min(std, self.actor.initial_std))

    def save_model(self, path_name, file_name):
        os.makedirs(path_name, exist_ok=True)

        torch.save(
            self.actor.state_dict(), os.path.join(path_name, file_name + "actor.pth")
        )
        torch.save(
            self.critic.state_dict(), os.path.join(path_name, file_name + "critic.pth")
        )

        print("PPO model is saved.")

    def load_model(self, path_name, file_name):
        self.actor.load_state_dict(
            torch.load(os.path.join(path_name, file_name + "actor.pth"))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(path_name, file_name + "critic.pth"))
        )

        print("PPO model is loaded.")
