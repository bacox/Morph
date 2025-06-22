import importlib
import json
import logging
import math
import os
import time
import gc
from collections import deque, defaultdict
from random import Random

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node

import psutil
import os


def log_memory(prefix=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)  # Resident Set Size in MB
    logging.info(f"[{prefix}] Memory usage: {mem_mb:.2f} MB")

class DissDL(Node):
    """
    This class defines the node on overlay graph

    """

    def save_plot(self, l, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    # def compute_my_flat_tensor(self):
    #     """
    #     Flatten and cache the current model parameters for similarity computation.
    #     """
    #     flat_params = []
    #     with torch.no_grad():
    #         for name, param in self.model.named_parameters():
    #             if param.requires_grad:
    #                 flat_params.append(param.detach().flatten().to(self.device))
    #     my_flat = torch.cat(flat_params)
    #     norm = my_flat.norm(p=2)
    #     self._cached_my_flat = my_flat / (norm + 1e-8)

    def compute_similarity(self, peer, fallback):
        # Check if model weights exist
        if peer not in self.peer_weights:
            # If we had computed similarity earlier, use it
            if peer in self.similarity_cache:
                return self.similarity_cache[peer]
            # Else return default
            logging.info(f"Data for peer {peer} do not exist even though they should.")
            return fallback

        peer_model_state = self.sharing.deserialized_model(self.peer_weights[peer])
        sim_scores = []

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                peer_tensor = peer_model_state.get(name)
                if peer_tensor is None:
                    continue

                my_tensor = param.detach().flatten().to(self.device)
                peer_tensor = peer_tensor.detach().flatten().to(self.device)

                if my_tensor.numel() != peer_tensor.numel():
                    continue

                my_norm = my_tensor.norm(p=2)
                peer_norm = peer_tensor.norm(p=2)
                if my_norm == 0 or peer_norm == 0:
                    continue

                my_tensor = my_tensor / my_norm
                peer_tensor = peer_tensor / peer_norm

                sim = F.cosine_similarity(my_tensor, peer_tensor, dim=0).item()

                sim_scores.append(sim)

        del peer_model_state  # Free memory explicitly

        if not sim_scores:
            return -1.0

        similarity = sum(sim_scores) / len(sim_scores)
        self.similarity_cache[peer] = similarity
        return sum(sim_scores) / len(sim_scores)


    def estimate_similarity(self, peer, fallback):
        """
        Estimate similarity to a peer we haven't received weights from,
        using transitive similarity via known peers.
        """
        if peer in self.sim_estimates_per_peer and self.sim_estimates_per_peer[peer]:
            estimates = []
            for _, intermediate_peer, sim_yz in self.sim_estimates_per_peer[peer]:
                sim_iy = self.similarity_cache.get(intermediate_peer)
                if sim_iy is None:
                    sim_iy = self.compute_similarity(intermediate_peer, fallback)
                estimates.append(sim_iy * sim_yz)

            if estimates:
                avg_est = sum(estimates) / len(estimates)
                self.similarity_cache[peer] = avg_est
                return avg_est

        # Fallback: use cached value or default
        if peer in self.similarity_cache:
            return self.similarity_cache[peer]

        return fallback



    def update_wanted_senders(self):
        if self.iteration == 0 or self.iteration % self.change_iter != 0:
            return

        current_senders = set(self.wanted_senders)
        candidate_senders = set(current_senders)

        possible_peers = {
            p for p in self.known_nodes
            if p != self.uid and self.connection_state.get(p, "NONE") in {"NONE", "ESTABLISHED"}
        }

        add_candidates = list(possible_peers - current_senders)
        remove_candidates = list(current_senders)

        # Need at least one to remove, and one to add
        if len(remove_candidates) <= 1 or not add_candidates:
            logging.info(f"Number of remove candidates: {len(remove_candidates)}. Can't remove.")
            return

        fallback_sim = (
            sum(self.similarity_cache.values()) / len(self.similarity_cache)
            if self.similarity_cache else 0.0
        )

        # Add logic
        add_scores = {}
        for peer in add_candidates:
            if peer in self.has_real_model:
                score = self.compute_similarity(peer, fallback_sim)
            else:
                score = self.estimate_similarity(peer, fallback_sim)
            add_scores[peer] = -score  # Prefer adding dissimilar peers
        peer_ids, values = zip(*add_scores.items())
        add_probs = torch.nn.functional.softmax(self.beta * torch.tensor(values, dtype=torch.float32), dim=0).numpy()
        peer_to_add = self.rng.choices(peer_ids, weights=add_probs, k=1)[0]
        candidate_senders.add(peer_to_add)
        logging.info(f"[Add] {[(pid, round(s, 4), round(p, 4)) for pid, s, p in zip(peer_ids, values, add_probs)]}")

        # Remove logic
        remove_scores = {}
        for peer in remove_candidates:
            if peer in self.has_real_model:
                score = self.compute_similarity(peer, fallback_sim)
            else:
                score = self.estimate_similarity(peer, fallback_sim)
            remove_scores[peer] = score  # Prefer removing similar peers
        peer_ids, values = zip(*remove_scores.items())
        remove_probs = torch.nn.functional.softmax(self.beta * torch.tensor(values, dtype=torch.float32), dim=0).numpy()
        peer_to_remove = self.rng.choices(peer_ids, weights=remove_probs, k=1)[0]
        candidate_senders.remove(peer_to_remove)
        logging.info(f"[Remove] {[(pid, round(-s, 4), round(p, 4)) for pid, s, p in zip(peer_ids, values, remove_probs)]}")

        self.wanted_senders = candidate_senders
        self.recent_peer_changes.append(peer_to_add)
        self.recent_peer_changes.append(peer_to_remove)

        if peer_to_remove in self.peer_weights:
            del self.peer_weights[peer_to_remove]

        # Establish connections for new sender
        if peer_to_add not in self.connected_peers:
            self.communication.init_connection(peer_to_add)
            self.communication.send(peer_to_add, {
                "CHANNEL": "CONNECT",
                "FROM": self.uid,
                "MESSAGE": "SYN"
            })
            self.connection_state[peer_to_add] = "SYN_SENT"
            self.wanted_senders.discard(peer_to_add)

        logging.debug(
            f"Iteration {self.iteration}: added={peer_to_add}, removed={peer_to_remove}, wanted_senders={sorted(self.wanted_senders)}"
        )



    def receive_DPSGD(self):
        return self.receive_channel("DPSGD", block=True)

    def receive_DPSGD_REQ(self):
        return self.receive_channel("DPSGD_REQ", block=True)

    def received_from_all(self, waiting_for):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in waiting_for:
            if (
                    (k not in self.peer_intents)
                    or len(self.peer_intents[k]) == 0
                    or self.peer_intents[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def received_from_wanted_senders(self):
        """
        Check if all wanted nodes have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.wanted_senders:
            if (
                    (k not in self.peer_payloads)
                    or len(self.peer_payloads[k]) == 0
                    or self.peer_payloads[k][0]["iteration"] != self.iteration
            ):
                return False
        return True


    def connect_neighbors(self):
        """
        Connects all neighbors. Sends HELLO. Waits for HELLO.
        Caches any data received while waiting for HELLOs.

        Raises
        ------
        RuntimeError
            If received BYE while waiting for HELLO

        """
        wait_acknowledgements = []
        for neighbor in self.connected_peers:
            if not self.communication.already_connected(neighbor):
                self.connect_neighbor(neighbor)
                wait_acknowledgements.append(neighbor)

        for neighbor in wait_acknowledgements:
            self.wait_for_hello(neighbor)

    def _handle_syn(self, sender, data):
        logging.debug(f"[{self.uid}] Received SYN from {sender}")
        if not self.communication.already_connected(sender):
            self.communication.init_connection(sender)

        self.communication.send(sender, {
            "CHANNEL": "CONNECT",
            "FROM": self.uid,
            "MESSAGE": "SYN-ACK",
            "iteration": self.iteration
        })
        self.connection_state[sender] = "SYN_RECEIVED"
        self.partial_connections.add(sender)

    def _handle_syn_ack(self, sender, data):
        logging.debug(f"[{self.uid}] Received SYN-ACK from {sender}")
        self.communication.send(sender, {
            "CHANNEL": "CONNECT",
            "FROM": self.uid,
            "MESSAGE": "ACK",
            "iteration": self.iteration
        })
        logging.debug(f"[{self.uid}] Sent ACK to {sender}.")
        sender_iter = data.get("iteration", 0)
        if sender_iter <= self.iteration:
            self.connection_state[sender] = "ESTABLISHED"
            self.connected_peers.add(sender)
            self.wanted_senders.add(sender)
            logging.debug(f"[{self.uid}] Added {sender} to connected_peers.")
        else:
            # Delay establishment
            self.delayed_syn_acks[sender] = sender_iter

    def _handle_ack(self, sender, data):
        logging.debug(f"[{self.uid}] Received ACK from {sender}.")
        sender_iter = data.get("iteration", 0)
        if sender_iter <= self.iteration:
            self.connection_state[sender] = "ESTABLISHED"
            self.connected_peers.add(sender)
            self.partial_connections.discard(sender)
            logging.debug(f"[{self.uid}] Added {sender} to connected_peers")
        else:
            # Delay establishment
            self.delayed_acks[sender] = sender_iter



    def run(self):
        logging.warning(f"[{self.uid}] Initial known_nodes: {self.known_nodes}")
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1
        self.rng = Random()
        self.rng.seed(self.dataset.random_seed + self.uid)

        self.connect_neighbors()
        logging.info("Connected to all neighbors")

        # Setup neighbors log file once
        self.neighbor_log_file = os.path.join(self.log_dir, f"neighbors_{self.uid}.txt")
        with open(self.neighbor_log_file, "w") as f:
            f.write(f"# Neighbor log for node {self.uid}\n")


        logging.info("Total number of neighbor: {}".format(len(self.my_neighbors)))

        for iteration in range(self.iterations):
            # log_memory(prefix=f"At iteration {iteration}")
            self.iteration = iteration
            if self.iteration == 0:
                class_counts = [0] * 10
                for _, label in self.dataset.trainset:
                    class_counts[label] += 1

                dist_log_path = os.path.join(self.log_dir, f"data_dist_node_{self.uid}.csv")
                with open(dist_log_path, "w") as f:
                    f.write("class,count\n")
                    for i, count in enumerate(class_counts):
                        f.write(f"{i},{count}\n")

            # Only remove old intents, keep future ones
            for sender in list(self.peer_intents):
                intents = self.peer_intents[sender]
                while intents and intents[0]["iteration"] < iteration:
                    logging.debug(f"Intent for {sender} for iteration {intents[0]['iteration']}, while in iteration {iteration}")
                    intents.popleft()
                if not intents:
                    del self.peer_intents[sender]

            for sender in list(self.peer_payloads):
                deque_x = self.peer_payloads[sender]
                # Drop stale messages (older than current iteration)
                while deque_x and deque_x[0]["iteration"] < iteration:
                    logging.debug(f"Payload for {sender} for iteration {deque_x[0]['iteration']}, while in iteration {iteration}")
                    deque_x.popleft()
                if not deque_x:
                    del self.peer_payloads[sender]


            # Refresh nodes_requesting_from_me from stored intents
            self.nodes_requesting_from_me.clear()
            for sender, intents in self.peer_intents.items():
                while intents and intents[0]["iteration"] < iteration:
                    intents.popleft()
                if intents and intents[0]["iteration"] == iteration:
                    if intents[0].get("want", False):
                        self.nodes_requesting_from_me.add(sender)

            # Local Phase
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            startup_delay = 1.0  # seconds
            if iteration == 0:
                time.sleep(startup_delay)

            # Promote delayed ACKs and SYN-ACKs when iteration catches up
            for peer, target_iter in list(self.delayed_acks.items()):
                if target_iter <= self.iteration:
                    self.connection_state[peer] = "ESTABLISHED"
                    self.connected_peers.add(peer)
                    self.partial_connections.discard(peer)
                    logging.debug(f"[{self.uid}] Finalized ACK connection with {peer} at iteration {self.iteration}")
                    del self.delayed_acks[peer]

            for peer, target_iter in list(self.delayed_syn_acks.items()):
                if target_iter <= self.iteration:
                    self.connection_state[peer] = "ESTABLISHED"
                    self.connected_peers.add(peer)
                    self.wanted_senders.add(peer)
                    logging.debug(f"[{self.uid}] Finalized SYN-ACK connection with {peer} at iteration {self.iteration}")
                    del self.delayed_syn_acks[peer]


            # Accept a_new dynamic connections
            while True:
                connect_msg = self.receive_channel("CONNECT", block=False)
                if connect_msg is None:
                    break
                sender, data = connect_msg
                message = data.get("MESSAGE")
                if message == "SYN":
                    self._handle_syn(sender, data)
                elif message == "SYN-ACK":
                    self._handle_syn_ack(sender, data)
                elif message == "ACK":
                    self._handle_ack(sender, data)
                else:
                    logging.warning(F"Unrecognised MESSAGE for CONNECT: {message}")

            self.trainer.train(self.dataset)

            # Update the graph - decide from whom to pull data
            self.update_wanted_senders()

            # Inform all known peers whether we are requesting data from them
            for peer in self.connected_peers | self.partial_connections:
                self.communication.send(peer, {
                    "CHANNEL": "DPSGD_REQ",
                    "iteration": self.iteration,
                    "want": peer in self.wanted_senders,
                })

            # Wait to receive request/intent messages from all neighbors
            logging.debug(f"[{self.uid}] Waiting for DPSGD_REQ from known nodes: {self.known_nodes}")
            waiting_for = set(self.connected_peers)
            while not self.received_from_all(waiting_for):
                logging.debug(f"Known nodes: {self.known_nodes}")
                msg = self.receive_DPSGD_REQ()
                if msg:
                    sender, data = msg
                    logging.debug(f"[{self.uid}] Received DPSGD_REQ from {sender}")

                    iter_num = data.get("iteration")
                    if iter_num > self.iteration:
                        if sender not in self.peer_intents:
                            self.peer_intents[sender] = deque()
                        self.peer_intents[sender].append(data)
                        continue
                    elif iter_num < self.iteration:
                        # discard stale message
                        continue

                    channel = data.get("CHANNEL")
                    if channel == "DPSGD_REQ":
                        if data.get("want", False):
                            self.nodes_requesting_from_me.add(sender)
                        self.known_nodes.add(sender)
                        if sender not in self.peer_intents:
                            self.peer_intents[sender] = deque()
                        self.peer_intents[sender].append(data)

            # Send similarities
            known_similarities = {
                p: self.similarity_cache[p]
                for p in self.known_nodes
                if (
                        p in self.similarity_cache and
                        self.similarity_cache[p] != 0
                )
            }
            to_send = self.sharing.get_data_to_send(degree=len(self.nodes_requesting_from_me))
            to_send["CHANNEL"] = "DPSGD"
            to_send["known_nodes"] = list(self.known_nodes)
            to_send["known_similarities"] = known_similarities

            # Send model only to nodes that requested it this round
            for peer in self.nodes_requesting_from_me | self.partial_connections:
                logging.debug("Sending model to requesting peer: %d", peer)
                self.communication.send(peer, to_send)

            logging.debug(f"Wanted senders: {self.wanted_senders}")
            while not self.received_from_wanted_senders():
                response = self.receive_DPSGD()
                if response:
                    sender, data = response

                    first_time_seen = sender not in self.has_real_model
                    self.peer_weights[sender] = {"params": data["params"]}
                    self.peer_model_counts[sender] += 1

                    if first_time_seen:
                        self.has_real_model.add(sender)
                        self.sim_estimates_per_peer.pop(sender, None)

                    logging.debug(
                        "Received Model from {} of iteration {}: {}".format(
                            sender,
                            data["iteration"],
                            "NotWorking" if "NotWorking" in data else "",
                        )
                    )
                    if "known_nodes" in data:
                        self.known_nodes.update(data["known_nodes"])
                    if sender not in self.peer_payloads:
                        self.peer_payloads[sender] = deque()

                    if data["iteration"] > self.iteration:
                        self.peer_payloads[sender].append(data)

                        if self.iteration > 0 and self.iteration % (self.change_iter - 1) == 0:
                            sim_map = data.get("known_similarities", {})
                            for target_peer, sim in sim_map.items():
                                if target_peer not in self.has_real_model:
                                    self.sim_estimates_per_peer[target_peer].append(
                                        (self.iteration, sender, sim)
                                    )
                        continue
                    elif data["iteration"] < self.iteration:
                        continue
                    else:
                        self.peer_payloads[sender].appendleft(data)
                        if self.iteration > 0 and self.iteration % (self.change_iter - 1) == 0:
                            sim_map = data.get("known_similarities", {})
                            for target_peer, sim in sim_map.items():
                                if target_peer not in self.has_real_model:
                                    self.sim_estimates_per_peer[target_peer].append(
                                        (self.iteration, sender, sim)
                                    )

            averaging_deque = dict()
            atleast_one = False
            for x, deque_x in self.peer_payloads.items():
                # Drop any old messages
                while deque_x and deque_x[0]["iteration"] < self.iteration:
                    deque_x.popleft()

                if len(deque_x) > 0:
                    this_message = deque_x[0]
                    if (
                            this_message["iteration"] == self.iteration
                            and "NotWorking" not in this_message
                    ):
                        averaging_deque[x] = deque_x
                        atleast_one = True
                    elif this_message["iteration"] == self.iteration:
                        deque_x.popleft()
                        logging.debug(
                            "Discarding message from {} of iteration {}".format(
                                x, this_message["iteration"]
                            )
                        )


            if atleast_one:
                self.sharing._averaging(averaging_deque)
            else:
                self.sharing.communication_round += 1

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
                with open(
                        os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                        "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "received_this_round": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                    ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                    ] = self.communication.total_data
            if hasattr(self.communication, "received_this_round"):
                results_dict["received_this_round"][
                    iteration + 1
                    ] = self.communication.received_this_round

            if rounds_to_train_evaluate == 0:


                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)

            with open(self.neighbor_log_file, "a") as f:
                peers = sorted(self.wanted_senders)
                peers_str = ", ".join(str(n) for n in peers)
                f.write(f"{self.iteration}: {peers_str}\n")

            with open(os.path.join(self.log_dir, f"{self.uid}_peer_model_counts_log.txt"), "a") as f:
                counts_str = ", ".join(f"{pid}:{cnt}" for pid, cnt in sorted(self.peer_model_counts.items()))
                f.write(f"{self.iteration}: {counts_str}\n")


            # with open(self.neighbor_log_file, "a") as f:
            #     connected_str = ", ".join(str(n) for n in sorted(self.connected_peers))
            #     neighbors_str = ", ".join(str(n) for n in sorted(self.wanted_senders))
            #     f.write(f"Iteration {iteration + 1} connected: {connected_str}\n")
            #     f.write(f"Iteration {iteration + 1} receiving from: {neighbors_str}\n")

            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()


        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")

    def cache_fields(
            self,
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )

    def instantiate(
            self,
            rank: int,
            machine_id: int,
            mapping: Mapping,
            graph: Graph,
            config,
            iterations=1,
            log_dir=".",
            weights_store_dir=".",
            log_level=logging.INFO,
            test_after=5,
            train_evaluate_after=1,
            reset_optimizer=1,
            *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_model(config["DATASET"], self.device)
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"], self.device)
        self.init_comm(config["COMMUNICATION"])

        self.prev_eval_loss = None
        self.prev_eval_accuracy = None
        self.peer_contribution_score = dict()
        self.prev_peer_activity_count = defaultdict(int)
        self.peer_activity_count = defaultdict(int)
        self.peer_weights = dict()
        # self._cached_my_flat = None
        self.recent_peer_changes = deque(maxlen=5)

        self.connection_state = defaultdict(lambda: "NONE")  # NONE, SYN_SENT, SYN_RECEIVED, SYN_ACKED, ESTABLISHED
        self.partial_connections = set()  # List of nodes we can send *intent*, but not yet "connected"

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)
        self.wanted_senders = set(self.graph.outgoing_neighbors(self.uid))
        self.connected_peers = set(self.my_neighbors)
        self.nodes_requesting_from_me = set()
        self.known_nodes = set(self.my_neighbors)

        self.init_sharing(config["SHARING"], self.device)
        self.peer_intents = dict()      # For DPSGD_REQ messages
        self.peer_payloads = dict()     # For DPSGD model payloads

        self.similarity_cache = dict()  # {peer_id: similarity_value}
        self.sim_estimates_per_peer = defaultdict(lambda: deque(maxlen=5))
        self.has_real_model = set()  # set of peers we've seen real weights from

        self.delayed_acks = dict()
        self.delayed_syn_acks = dict()

        self.peer_model_counts = defaultdict(int)

        self.connect_neighbors()

    def __init__(
            self,
            rank: int,
            machine_id: int,
            mapping: Mapping,
            graph: Graph,
            config,
            iterations=1,
            log_dir=".",
            weights_store_dir=".",
            log_level=logging.INFO,
            test_after=5,
            train_evaluate_after=1,
            reset_optimizer=1,
            *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.procs_per_machine), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        # torch.set_num_threads(1)

        use_cuda = config.get("CUDA", {}).get("use_cuda", False)
        if isinstance(use_cuda, str):
            use_cuda = use_cuda.lower() == "true"

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        # torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            *args
        )

        nodeConfigs = config["VARS"]
        self.change_iter = (
            nodeConfigs["change_topology_iter"] if "change_topology_iter" in nodeConfigs else 5
        )
        self.beta = float(nodeConfigs["beta"]) if "beta" in nodeConfigs else 1.0

        self.model.to(self.device)

        logging.debug(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()