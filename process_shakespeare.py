from pathlib import Path
from typing import List, Optional
import numpy as np
import json
from flwr_datasets.preprocessor.divider import Divider


def get_client_data(client_id, partitions, partitions_per_client) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for pid in partitions_per_client[client_id]:
        partition = partitions.load_partition(partition_id=pid, split="train")
        partition_np = partition.with_format("numpy")

        X, y = partition_np["x"], partition_np["character_id"]
        X_list.append(X)
        y_list.append(y)
    # Concatenate into single numpy arrays
    X_client = np.concatenate(X_list, axis=0)
    y_client = np.concatenate(y_list, axis=0)
    return X_client, y_client


def client_load_shakespeare(c_id: int, num_clients=100, alpha=0.1, partition_by="character_id"):
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner, DirichletPartitioner

    fds = FederatedDataset(
        # dataset="flwrlabs/shakespeare",
        # partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
        dataset="flwrlabs/shakespeare",
        partitioners={"train": DirichletPartitioner(num_partitions=100, alpha=0.1, partition_by="character_id")},
    )
    NUM_PARTITIONS = fds._partitioners["train"].num_partitions
    # Assign partitions to clients (round-robin)
    partitions_per_client = [[] for _ in range(num_clients)]

    for pid in range(NUM_PARTITIONS):
        client_id = pid % num_clients
        partitions_per_client[client_id].append(pid)

    return get_client_data(c_id, fds, partitions_per_client)


def load_shakespeare_range(
    num_clients=100, alpha=0.1, partition_by="character_id", select_clients: Optional[List] = None
) -> List[tuple[np.ndarray, np.ndarray]]:
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner, DirichletPartitioner

    fds = FederatedDataset(
        # dataset="flwrlabs/shakespeare",
        # partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
        dataset="flwrlabs/shakespeare",
        partitioners={"train": DirichletPartitioner(num_partitions=100, alpha=0.1, partition_by="character_id")},
    )
    NUM_PARTITIONS = fds._partitioners["train"].num_partitions
    # Assign partitions to clients (round-robin)
    partitions_per_client = [[] for _ in range(num_clients)]

    for pid in range(NUM_PARTITIONS):
        client_id = pid % num_clients
        partitions_per_client[client_id].append(pid)

    if select_clients is not None:
        return [get_client_data(i, fds, partitions_per_client) for i in select_clients]
    return [get_client_data(i, fds, partitions_per_client) for i in range(num_clients)]


def try_flwr():
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner, DirichletPartitioner

    # properties: ['image', 'writer_id', 'hsf_id', 'character']
    # fds = FederatedDataset(
    #     # dataset="flwrlabs/shakespeare",
    #     # partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
    #     dataset="flwrlabs/shakespeare",
    #     partitioners={"train": DirichletPartitioner(num_partitions=100, alpha=0.1, partition_by="character")},
    # )
    from flwr_datasets.utils import divide_dataset

    fds = FederatedDataset(
        dataset="flwrlabs/shakespeare",
        partitioners={"train": 100},
    )
    part0 = fds.load_partition(partition_id=0)
    train_new, test_new = divide_dataset(part0, division=[0.9, 0.1])
    Path("./tmp/shakespeare").mkdir(parents=True, exist_ok=True)

    train_new.save_to_disk("./tmp/shakespeare/train")
    test_new.save_to_disk("./tmp/shakespeare/test")

    Path("./eval/data/shakespeare/").mkdir(exist_ok=True, parents=True)
    # exit()
    # print(new_fds)
    # # Save dataset to disk
    # fds.load_split("train").save_to_disk("./tmp/shakespeare")
    # fds.load_split("test").save_to_disk("./tmp/shakespeare")
    # exit()
    partition = fds.load_partition(partition_id=0)

    # print(fds._partitioners["train"].num_partitions)  # number of clients
    NUM_PARTITIONS = fds._partitioners["train"].num_partitions
    NUM_CLIENTS = 100

    partition_np = partition.with_format("numpy")
    print(f"keys: {partition_np}")
    # exit()

    X_train, y_train = partition_np["x"], partition_np["character_id"]

    print(X_train.shape, y_train.shape)

    print(np.unique(y_train, return_counts=True))

    # Assign partitions to clients (round-robin)
    partitions_per_client = [[] for _ in range(NUM_CLIENTS)]
    for pid in range(NUM_PARTITIONS):
        client_id = pid % NUM_CLIENTS
        partitions_per_client[client_id].append(pid)

    # print(partitions_per_client)
    # Print first 10 clients and their assigned partitions

    def get_client_data(client_id):
        X_list, y_list = [], []
        # print(f"Partition keys: {list(partitions_per_client)}")
        # print(
        #     f"Num partitions per client_id: {client_id} = {len(partitions_per_client[client_id])} of total: {len(partitions_per_client)}"
        # )
        for pid in partitions_per_client[client_id]:
            partition = fds.load_partition(partition_id=pid, split="train")
            partition_np = partition.with_format("numpy")

            X, y = partition_np["x"], partition_np["character_id"]
            X_list.append(X)
            y_list.append(y)
        # Concatenate into single numpy arrays
        X_client = np.concatenate(X_list, axis=0)
        y_client = np.concatenate(y_list, axis=0)
        return X_client, y_client

    dfs = []
    import pandas as pd

    for client_id in tqdm(range(NUM_CLIENTS)):
        # print(f"Client {client_id}: Partitions {partitions_per_client[client_id]}")
        X_client, y_client = get_client_data(client_id)
        # print(f"Client {client_id}: Data shape {X_client.shape}, Labels shape {y_client.shape}")
        # print(np.unique(y_client, return_counts=True))

        # Save the count of samples per client per character in a dataframe

        df = pd.DataFrame(np.unique(y_client, return_counts=True)).T
        df.columns = ["character_id", "num_samples"]
        df["client_id"] = client_id
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    # print(df_all)
    # plot histogram of num_samples
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Random sample 5 clients to plot
    sampled_df = df_all[df_all["client_id"].isin(np.random.choice(NUM_CLIENTS, size=5, replace=False))]

    plt.figure(figsize=(30, 6))
    # Plot count of characters per client with hue as client_id
    sns.barplot(data=sampled_df, x="character_id", y="num_samples", hue="client_id", ci=None)

    plt.title("Distribution of Number of Samples per Client")
    plt.xlabel("Number of Samples")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("./eval/data/shakespeare/num_samples_per_client_histogram.png")
    plt.show()

    # Plot just the total number of samples per client
    df_total = df_all.groupby("client_id")["num_samples"].sum().reset_index()
    print(df_total)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_total, x="client_id", y="num_samples", ci=None)
    plt.title("Total Number of Samples per Client")
    plt.xlabel("Client ID")
    plt.ylabel("Total Number of Samples")
    plt.grid()
    plt.savefig("./eval/data/shakespeare/total_num_samples_per_client.png")
    plt.show()

    # Plot histogram of total number of samples per client of all clients
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_all, x="num_samples", bins=30, kde=True)
    plt.title("Histogram of Total Number of Samples per Client")
    plt.xlabel("Total Number of Samples")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("./eval/data/shakespeare/total_num_samples_per_client_histogram.png")
    plt.show()


def preprocess_shakespeare():
    data_type = ["train", "test"]
    Path("./eval/data/shakespeare").mkdir(exist_ok=True, parents=True)
    print("preprocessing shakespeare dataset...")
    for data_t in data_type:
        path = f"./eval/data/shakespeare/{data_t}"

        users = {}
        for f in Path(path).iterdir():
            if f.stem.endswith("_split"):
                continue
            print(f"processing {f}...")

            with open(f, "r") as inf:
                sample_file = json.load(inf)
                print(f"Sample file: {sample_file.keys()}")

                sample_file_users = sample_file["users"]
                for idx, file_user in enumerate(sample_file_users):
                    obj = {
                        "users": [file_user],
                        "user_data": {file_user: sample_file["user_data"][file_user]},
                        "num_samples": [sample_file["num_samples"][idx]],
                    }

                    file_path = f.parent / f"{f.stem}_{idx}_split.json"
                    # print(file_path)
                    with open(file_path, "w+") as out_file:
                        json.dump(obj, out_file)
                sample_file_num_samples = sample_file["num_samples"]
                users.update({u: sample_file_num_samples[i] for i, u in enumerate(sample_file_users)})

        print(f"[{data_t}] number of users: {len(users)}")
        print(f"[{data_t}] total number of samples: {sum(users.values())}")
        print(f"[{data_t}] average number of samples per user: {np.mean(list(users.values()))}")
        print(f"[{data_t}] std of number of samples per user: {np.std(list(users.values()))}")
        print(f"[{data_t}] max number of samples per user: {np.max(list(users.values()))}")
        print(f"[{data_t}] min number of samples per user: {np.min(list(users.values()))}")
        print(f"[{data_t}] median number of samples per user: {np.median(list(users.values()))}")

        # save user distribution as json
        with open(f"./eval/data/shakespeare/user_distribution_{data_t}.json", "w") as outf:
            json.dump(users, outf)
        print(f"user distribution saved to ./eval/data/shakespeare/user_distribution_{data_t}.json")


if __name__ == "__main__":
    from tqdm import tqdm

    # preprocess_shakespeare()
    preprocess_shakespeare()
    # try_flwr()
