from copy import deepcopy
from typing import Any, List
import sys
import os

# sys.path.append("../src")

from pyIAAS import *


def search_datasets(dataset_list, data_dir, target_name, test_ratio):
    """
    search network structure using IAAS framework
    @return:
    """
    set_seed(42)
    print(dataset_list)
    cfg = Config(config_file)
    if debug:
        cfg.NASConfig['IterationEachTime'] = 1  # quick code test
        cfg.NASConfig['EPISODE'] = 3
    for dataset in dataset_list:
        task_name = dataset.split(".")[0]
        task_cfg = deepcopy(cfg)
        task_cfg.NASConfig["OUT_DIR"] = os.path.join('IAAS', task_name)
        try:
            run_search(
                task_cfg, os.path.join(data_dir, dataset), target_name, test_ratio
            )
            with open("main log.txt", "a") as f:
                f.writelines([f"successfully finish task IAAS {task_name}\n"])
            # if debug:
            #     exit(0)
        except Exception as e:
            with open("main log.txt", "a") as f:
                f.writelines([f"error at task IAAS {task_name}:\n{e}\n"])


def search_datasets_no_pool(dataset_list, data_dir, target_name, test_ratio):
    """
    search network structure using IAAS framework, ablation study, no pool
    @return:
    """
    set_seed(44)
    print(dataset_list)
    cfg = Config(config_file)
    if debug:
        cfg.NASConfig['IterationEachTime'] = 1  # quick code test
        cfg.NASConfig['EPISODE'] = 3

    for dataset in dataset_list:
        task_name = dataset.split(".")[0]
        task_cfg = deepcopy(cfg)
        task_cfg.NASConfig["OUT_DIR"] = os.path.join('IAAS_n', task_name)
        task_cfg.NASConfig["RandomAddNumber"] = 0
        task_cfg.NASConfig["KeepPrevNet"] = False
        task_cfg.NASConfig["NetPoolSize"] = 1
        try:
            run_search(
                task_cfg, os.path.join(data_dir, dataset), target_name, test_ratio
            )
            with open("main log IAAS_n.txt", "a") as f:
                f.writelines([f"successfully finish task IAAS_n {task_name}\n"])
        except Exception as e:
            with open("main log.txt", "a") as f:
                f.writelines([f"error at task IAAS_n {task_name}:\n{e}\n"])


def search_datasets_no_pool_no_selector(dataset_list, data_dir, target_name, test_ratio):
    """
    search network structure using IAAS framework, ablation study, no pool no selector
    @return:
    """
    set_seed(44)
    print(dataset_list)
    cfg = Config(config_file)
    if debug:
        cfg.NASConfig['IterationEachTime'] = 1  # quick code test
        cfg.NASConfig['EPISODE'] = 3

    for dataset in dataset_list:
        task_name = dataset.split(".")[0]
        task_cfg = deepcopy(cfg)
        task_cfg.NASConfig["OUT_DIR"] = os.path.join('IAAS_sn', task_name)
        task_cfg.NASConfig["RandomAddNumber"] = 0
        task_cfg.NASConfig["KeepPrevNet"] = False
        task_cfg.NASConfig["NetPoolSize"] = 1
        try:
            run_no_selector_search(
                task_cfg, os.path.join(data_dir, dataset), target_name, test_ratio
            )
            with open("main log IAAS_sn.txt", "a") as f:
                f.writelines([f"successfully finish task IAAS_sn {task_name}\n"])
        except Exception as e:
            with open("main log.txt", "a") as f:
                f.writelines([f"error at task IAAS_sn {task_name}:\n{e}\n"])


def search_datasets_no_selector(dataset_list, data_dir, target_name, test_ratio):
    """
    search network structure using IAAS framework, ablation study, no selector
    @return:
    """
    set_seed(44)
    print(dataset_list)
    cfg = Config(config_file)
    if debug:
        cfg.NASConfig['IterationEachTime'] = 1  # quick code test
        cfg.NASConfig['EPISODE'] = 3
    for dataset in dataset_list:
        task_name = dataset.split(".")[0]
        task_cfg = deepcopy(cfg)
        task_cfg.NASConfig["OUT_DIR"] = os.path.join('IAAS_s', task_name)
        try:
            run_no_selector_search(
                task_cfg, os.path.join(data_dir, dataset), target_name, test_ratio
            )
            with open("main log IAAS_s.txt", "a") as f:
                f.writelines([f"successfully finish task IAAS_s {task_name}\n"])
        except Exception as e:
            with open("main log.txt", "a") as f:
                f.writelines([f"error at task IAAS_s {task_name}:\n{e}\n"])


def run_random_search(cfg, input_file, target_name, test_ratio):
    cache_dir = "cache"
    os.makedirs(cfg.NASConfig["OUT_DIR"], exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    # process data and store middle file in cache dir
    x, y = get_data(
        cache_dir,
        input_file,
        target_name,
        cfg.NASConfig.timeLength,
        cfg.NASConfig.predictLength,
    )

    # preprocess data by splitting train test datasets then convert to torch.Tensor object
    data = train_test_split(x, y, test_ratio)
    data = [torch.tensor(i, dtype=torch.float) for i in data]
    logger_ = get_logger(f"random search", cfg.LOG_FILE)
    env_ = RandomSearchEnv.try_load(cfg, logger_)
    if env_ is None:
        env_ = RandomSearchEnv(cfg, cfg.NASConfig["NetPoolSize"], data)
        states = env_.reset()
    else:
        states = env_.get_state()
    try_load_rng_state(cfg, logger_)
    st = time.time()
    for i in range(cfg.NASConfig["EPISODE"]):
        env_.save()
        save_rng_states(cfg, logger_)
        env_.step()
        logger_.critical(
            f"episode {i} finish,\tpool {len(env_.net_pool)},\tperformance:{env_.performance()}\ttop performance:{env_.top_performance()}"
        )
    logger_.critical(
        f'Search episode: {cfg.NASConfig["EPISODE"]}\t Best performance: {env_.top_performance()}\t Search time :{time.time() - st:.2f} seconds'
    )


def run_no_selector_search(cfg, input_file, target_name, test_ratio):
    """
    run experiment without selector, the selection is replaced by random action
    """
    cache_dir = "cache"
    os.makedirs(cfg.NASConfig["OUT_DIR"], exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    # process data and store middle file in cache dir
    x, y = get_data(
        cache_dir,
        input_file,
        target_name,
        cfg.NASConfig.timeLength,
        cfg.NASConfig.predictLength,
    )

    # preprocess data by splitting train test datasets then convert to torch.Tensor object
    data = train_test_split(x, y, test_ratio)
    data = [torch.tensor(i, dtype=torch.float) for i in data]
    logger_ = get_logger(f"no selector", cfg.LOG_FILE)
    env_ = NasEnv.try_load(cfg, logger_)
    if env_ is None:
        env_ = NasEnv(cfg, cfg.NASConfig["NetPoolSize"], data)
        states = env_.reset()
    else:
        states = env_.get_state()
    agent_ = Agent.try_load(cfg, logger_)
    if agent_ is None:
        agent_ = Agent(cfg, 16, 50, cfg.NASConfig["MaxLayers"])
    replay_memory = ReplayMemory()
    replay_memory.load_memories(cfg.NASConfig["OUT_DIR"])
    try_load_rng_state(cfg, logger_)
    st = time.time()
    action_set = [
        SelectorActorNet.UNCHANGE,
        SelectorActorNet.WIDER,
        SelectorActorNet.DEEPER,
        SelectorActorNet.PRUNE,
    ]
    action_set = [torch.tensor(i, dtype=torch.int64) for i in action_set]
    for i in range(cfg.NASConfig["EPISODE"]):
        agent_.save()
        env_.save()
        save_rng_states(cfg, logger_)
        action = agent_.get_action(states)
        # replace selector action with random action
        logger_.critical(
            f'original action:{[["UNCHANGE", "WIDER", "DEEPER", "PRUNE", ][i["select"].item()] for i in action["action"]]}')
        for j in range(len(action["action"])):
            action["action"][j]["select"] = action_set[torch.randint(0, 4, (1,))].to(
                action["action"][j]["select"].device
            )
        # print random action
        logger_.critical(
            f'random action:{[["UNCHANGE", "WIDER", "DEEPER", "PRUNE", ][i["select"].item()] for i in action["action"]]}')
        states, in_pool_trajectory, finished_trajectory = env_.step(action)
        replay_memory.record_trajectory(in_pool_trajectory, finished_trajectory)
        replay_memory.save_memories(cfg.NASConfig["OUT_DIR"])
        agent_.update(replay_memory)
        record_pool_statistic(cfg, env_)
        logger_.critical(
            f"episode {i} finish,\tpool {len(env_.net_pool)},\tperformance:{env_.performance()}\ttop performance:{env_.top_performance()}"
        )
    logger_.critical(
        f'Search episode: {cfg.NASConfig["EPISODE"]}\t Best performance: {env_.top_performance()}\t Search time :{time.time() - st:.2f} seconds'
    )


def search_datasets_random_search(dataset_list, data_dir, target_name, test_ratio):
    """
    search network structure using random search method with netpool
    @return:
    """
    set_seed(44)
    print(dataset_list)
    cfg = Config(config_file)
    if debug:
        cfg.NASConfig['IterationEachTime'] = 1  # quick code test
        cfg.NASConfig['EPISODE'] = 3
    for dataset in dataset_list:
        task_name = dataset.split(".")[0]
        task_cfg = deepcopy(cfg)
        task_cfg.NASConfig["OUT_DIR"] = os.path.join('Random Search', task_name)
        run_random_search(
            task_cfg, os.path.join(data_dir, dataset), target_name, test_ratio
        )
        try:
            with open("random search log.txt", "a") as f:
                f.writelines([f"successfully finish task Random Search {task_name}\n"])
        except Exception as e:
            with open("random search log.txt", "a") as f:
                f.writelines([f"error at task Random Search {task_name}:\n{e}\n"])


class NoSelectorSearchEnv(NasEnv):
    def reset(self):
        reset_model_count()
        X_train, y_train, X_test, y_test = self.train_test_data
        self.net_pool = [
            generate_new_model_config(
                self.cfg, self.feature_shape, self.target_shape, [i]
            ).generate_model()
            for i in self.cfg.modulesConfig.keys()
        ]
        self._train_and_test()
        self.net_pool = sorted(self.net_pool, key=lambda x: x.test_loss)
        self.net_pool = self.net_pool[: self.pool_size]
        self.render()
        self.origin_net_pool = []
        return self.get_state()

    def step(self, action, action_type, transition):
        if action_type == "render":
            self.net_pool.extend(self.origin_net_pool)
            self.origin_net_pool = []
            self._train_and_test()
            self.net_pool = sorted(self.net_pool, key=lambda x: x.test_loss)
            reward_dict = self.get_reward()
            for i in transition:
                i["reward"] = reward_dict[i["next net"]]
                t = i["prev net"].state, i["action"], i["reward"], i["policy"]
                t = recursive_tensor_detach(t)
                i["next net"].transitions.append(Transition(*t))
            in_pool_trajectory, finished_trajectory = [], []
            for i in range(len(self.net_pool)):
                if i < self.pool_size:
                    in_pool_trajectory.append(self.net_pool[i].transitions)
                    self.net_pool[i].update_pool_state(True)
                else:
                    finished_trajectory.append(self.net_pool[i].transitions)
                    self.net_pool[i].update_pool_state(False)
            self.net_pool = self.net_pool[: self.pool_size]
            state = self.get_state()
            return state, in_pool_trajectory, finished_trajectory

        net_pool: List[NasModel] = []
        origin_net_pool: List[NasModel] = []
        prev_transitions = transition
        transition: List[dict] = []

        # remove all model from pool
        for net in self.net_pool:
            net.update_pool_state(False)

        for i in range(len(self.net_pool)):
            net = self.net_pool[i]
            action_i = action["action"][i]
            prev_net = net

            if self.cfg.NASConfig["KeepPrevNet"] and len(self.origin_net_pool) == 0:
                origin_net_pool.append(prev_net)
                transition_item = {}
                transition_item["prev net"] = prev_net
                transition_item["next net"] = prev_net
                transition_item["action"] = copy.deepcopy(action["action"][i])
                transition_item["action"]["select"] = SelectorActorNet.UNCHANGE
                transition_item["policy"] = action["policy"][i]
                transition.append(transition_item)

            # select representations
            if action_type == "deeper":
                if (
                        len(net.model_config.modules) < self.cfg.NASConfig["MaxLayers"]
                ):  # constrain the network's depth
                    self.logger.info(f"net index {i}-{net.index} :deeper the net {net}")
                    net = net.perform_deeper_transformation(action_i["deeper"])
            elif action_type == "wider":
                self.logger.info(f"net index {i}-{net.index} :wider the net {net}")
                net = net.perform_wider_transformation(action_i["wider"])

            net_pool.append(net)
            transition_item = {}
            transition_item["prev net"] = prev_net
            transition_item["next net"] = net
            transition_item["action"] = action["action"][i]
            transition_item["policy"] = action["policy"][i]
            transition.append(transition_item)
        if self.cfg.NASConfig["KeepPrevNet"] and len(self.origin_net_pool) == 0:
            self.origin_net_pool = origin_net_pool
        self.net_pool = net_pool
        for net in self.net_pool:
            net.update_pool_state(True)
        self.net_pool = list(set(self.net_pool))
        state = self.get_state()
        for i in transition:
            for j in prev_transitions:
                if j["next net"] == i["prev net"]:
                    j["next net"] = i["next net"]
        return state, transition


class RandomSearchEnv(NasEnv):
    def step(self):
        random_nets = self.generate_random_net()
        self.net_pool.extend(random_nets)
        self._train_and_test()
        self.net_pool = sorted(self.net_pool, key=lambda x: x.test_loss)
        self.net_pool = self.net_pool[: self.pool_size]

    def generate_random_net(self, net_number=None):
        net_number = len(self.net_pool) + 1
        random_nets = []
        for i in range(net_number):
            net_layers = min(
                max(1, int(random.expovariate(0.2))), self.cfg.NASConfig["MaxLayers"]
            )
            skeleton = [
                random.sample(self.cfg.modulesCls.keys(), 1)[0]
                for i in range(net_layers)
            ]
            random_model = generate_new_model_config(
                self.cfg, self.feature_shape, self.target_shape, skeleton
            ).generate_model()
            random_nets.append(random_model)
        return random_nets


if __name__ == "__main__":
    os.chdir('../data')  # change working dir
    debug = False # debug mode

    load_dataset_list = [
        "ME_spring.csv",
        "ME_summer.csv",
        "ME_autumn.csv",
        "ME_winter.csv",
        "NH_spring.csv",
        "NH_summer.csv",
        "NH_autumn.csv",
        "NH_winter.csv",
    ]

    wind_dataset_list = [
        "WF1_spring.csv",
        "WF1_summer.csv",
        "WF1_autumn.csv",
        "WF1_winter.csv",
        "WF2_spring.csv",
        "WF2_summer.csv",
        "WF2_autumn.csv",
        "WF2_winter.csv",
    ]

    # please delete output data generated by previous runs if you want to run and new search experiment
    # uncomment code blocks corresponding to the experiment you want to run

    config_file = 'NASConfig_wind.json'
    search_datasets(wind_dataset_list, 'wind dataset aligned', '实际功率', 24 * 5)
    config_file = 'NASConfig_load.json'
    search_datasets(load_dataset_list, 'load_datasets', 'RT_Demand', int(24 * (30 + 31 + 30) * 0.2))

    # config_file = 'NASConfig_wind.json'
    # search_datasets_random_search(wind_dataset_list, 'wind dataset aligned', '实际功率', 24 * 5)
    # config_file = 'NASConfig_load.json'
    # search_datasets_random_search(load_dataset_list, 'load_datasets', 'RT_Demand', int(24 * (30 + 31 + 30) * 0.2))

    # config_file = 'NASConfig_wind.json'
    # search_datasets_no_pool(wind_dataset_list, 'wind dataset aligned', '实际功率', 24 * 5)
    # config_file = 'NASConfig_load.json'
    # search_datasets_no_pool(load_dataset_list, 'load_datasets', 'RT_Demand', int(24 * (30 + 31 + 30) * 0.2))

    # config_file = "NASConfig_wind.json"
    # search_datasets_no_selector(
    #     wind_dataset_list, "wind dataset aligned", "实际功率", 24 * 5
    # )
    # config_file = "NASConfig_load.json"
    # search_datasets_no_selector(
    #     load_dataset_list, "load_datasets", "RT_Demand", int(24 * (30 + 31 + 30) * 0.2)
    # )

    # config_file = 'NASConfig_wind.json'
    # search_datasets_no_pool_no_selector(wind_dataset_list, 'wind dataset aligned', '实际功率', 24 * 5)
    # config_file = 'NASConfig_load.json'
    # search_datasets_no_pool_no_selector(load_dataset_list, 'load_datasets', 'RT_Demand', int(24 * (30 + 31 + 30) * 0.2))
