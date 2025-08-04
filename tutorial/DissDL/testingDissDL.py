import logging
from pathlib import Path
from shutil import copy

from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Linear import Linear
from decentralizepy.node.DissDL.DissDL import DissDL as BaseLine
from decentralizepy.node.DissDL.GlobalDataSimilarity import DissDL as GDS
from decentralizepy.node.DissDL.GlobalModelSimilarity import DissDL as GMS


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config


if __name__ == "__main__":
    args = utils.get_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    print(f"Path to config file: {args.config_file}")
    config = read_ini(args.config_file)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    copy(args.config_file, args.log_dir)
    copy(args.graph_file, args.log_dir)
    utils.write_args(args, args.log_dir)

    alg_name = my_config["ALGORITHM"]["name"]
    print(f"Algorithm: {alg_name}")

    # Match the algorithm name to the target class
    target_mapping = {
        "Baseline": BaseLine,
        "GDS": GDS,
        "GMS": GMS,
    }
    target_class = target_mapping.get(alg_name)
    if target_class is None:
        raise ValueError(f"Unknown algorithm name: {alg_name}. Supported algorithms: {list(target_mapping.keys())}")
    print(f"Using target class: {target_class}")
    g = Graph()
    g.read_graph_from_file(args.graph_file, args.graph_type)
    n_machines = args.machines
    procs_per_machine = args.procs_per_machine[0]

    l = Linear(n_machines, procs_per_machine)
    m_id = args.machine_id

    processes = []
    for r in range(procs_per_machine):
        processes.append(
            mp.Process(
                target=target_class,
                args=[
                    r,
                    m_id,
                    l,
                    g,
                    my_config,
                    args.iterations,
                    args.log_dir,
                    args.weights_store_dir,
                    log_level[args.log_level],
                    args.test_after,
                    args.train_evaluate_after,
                    args.reset_optimizer,
                ],
            )
        )

    for p in processes:
        p.start()

    for p in processes:
        p.join()
