import logging

import torch

from decentralizepy import utils


class Training:
    """
    This class implements the training module for a single node.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
        device=None,
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
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs : bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.

        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.rounds = utils.conditional_value(rounds, "", int(1))
        self.full_epochs = utils.conditional_value(full_epochs, "", False)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)
        self.device = device

    def reset_optimizer(self, optimizer):
        """
        Replace the current optimizer with a a_new one

        Parameters
        ----------
        optimizer : torch.optim
            A a_new optimizer

        """
        self.optimizer = optimizer

    def eval_loss(self, dataset):
        """
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        num_samples = len(trainset.dataset)
        num_counted = 0
        with torch.no_grad():

            for data, target in trainset:

                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
                num_counted += len(data)
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        logging.info("Number of samples counted: {}".format(num_counted))
        logging.info("Total number of samples from dataset: {}".format(num_samples))
        return loss

    def eval_loss_from_loader(self, loader):
        """
        Evaluate the loss on a given DataLoader (e.g., for validation or test).

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            A DataLoader providing batches of (data, target) tuples.

        Returns
        -------
        float
            Average loss over the entire loader.
        """
        self.model.eval()
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in loader:

                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
        loss = epoch_loss / count if count > 0 else float("inf")
        logging.info("Validation loss after iteration: {}".format(loss))
        return loss

    def eval_loss_and_accuracy_from_loader(self, loader):
        """
        Evaluate both loss and accuracy on a given DataLoader.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader

        Returns
        -------
        tuple(float, float)
            (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss_val = self.loss(output, target)
                total_loss += loss_val.item()

                pred = output.argmax(dim=1)
                total_correct += (pred == target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / len(loader) if len(loader) > 0 else float("inf")
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        logging.info(f"Eval — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def trainstep(self, data, target):
        """
        One training step on a minibatch.

        Parameters
        ----------
        data : any
            Data item
        target : any
            Label

        Returns
        -------
        int
            Loss Value for the step

        """
        self.model.zero_grad()

        data = data.to(self.device)
        target = target.to(self.device)

        output = self.model(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def train_full(self, dataset):
        """
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        for epoch in range(self.rounds):
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                logging.debug("Starting minibatch {} with num_samples: {}".format(count, len(data)))
                logging.debug("Classes: {}".format(target))
                epoch_loss += self.trainstep(data, target)
                count += 1
            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    def train(self, dataset):
        """
        One training iteration

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.train()

        if self.full_epochs:
            self.train_full(dataset)
        else:
            iter_loss = 0.0
            count = 0
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            while count < self.rounds:
                for data, target in trainset:
                    iter_loss += self.trainstep(data, target)
                    count += 1
                    logging.debug("Round: {} loss: {}".format(count, iter_loss / count))
                    if count >= self.rounds:
                        break
