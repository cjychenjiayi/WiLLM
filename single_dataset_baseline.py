import os
import torch
import torch.nn as nn
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.multiprocessing as mp

from models.basic_model import Model
from dataloader import load_data


# =====================================================
# Utils
# =====================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================
# Trainer
# =====================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

        print(f"Using device: {self.device}")

        self.output_path, self.log_file = self._setup_output()

        self.model, self.train_loader, self.test_loader = self._setup_data_and_model()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.best_test_acc = 0.0
        self.best_epoch = 0

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    # ----------------------------
    # Setup
    # ----------------------------

    def _setup_output(self):
        tag = f'{self.args.model}-bs_{self.args.batch_size}-lr_{self.args.lr}-seed_{self.args.seed}'
        output_path = os.path.join(self.args.output_path, tag)
        os.makedirs(output_path, exist_ok=True)
        log_file = os.path.join(output_path, "log.txt")
        return output_path, log_file

    def _rebuild_loader(self, loader, shuffle=False, sampler=None):
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,      # 单进程最稳
            pin_memory=False,
            persistent_workers=False
        )

    def _setup_data_and_model(self):

        train_loader, test_loader, param = load_data(
            self.args.dataset,
            self.args.data_root,
            self.args.batch_size
        )

        # 强制稳定 DataLoader
        train_loader = self._rebuild_loader(train_loader, shuffle=True)
        test_loader = self._rebuild_loader(test_loader, shuffle=False)

        # 子采样
        if self.args.train_ratio < 1.0:
            total_samples = len(train_loader.dataset)
            keep_samples = int(total_samples * self.args.train_ratio)
            indices = list(range(total_samples))
            random.shuffle(indices)
            subset_indices = indices[:keep_samples]

            sampler = SubsetRandomSampler(subset_indices)
            train_loader = self._rebuild_loader(train_loader, sampler=sampler)

        subcarry, timestamp, num_classes = param

        model = Model(
            num_classes=num_classes,
            model_type=self.args.model,
            subcarry=subcarry,
            timestamp=timestamp,
            **vars(self.args)
        ).to(self.device)

        return model, train_loader, test_loader

    # ----------------------------
    # Train / Eval
    # ----------------------------

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.train_loader, desc="Train", leave=False):
            inputs = inputs.to(self.device)
            labels = torch.argmax(labels.to(self.device), dim=1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Test", leave=False):
                inputs = inputs.to(self.device)
                labels = torch.argmax(labels.to(self.device), dim=1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    # ----------------------------
    # Training controller
    # ----------------------------

    def fit(self):
        for epoch in range(1, self.args.max_epoch + 1):

            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            self._log(
                f"Epoch {epoch}/{self.args.max_epoch} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f}"
            )

            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_epoch = epoch
                self.save_checkpoint()

        self._log(
            f"Best Test Accuracy: {self.best_test_acc:.4f} "
            f"at Epoch {self.best_epoch}"
        )

        self.plot_metrics()

    # ----------------------------
    # Utilities
    # ----------------------------

    def save_checkpoint(self):
        save_path = os.path.join(self.output_path, "best_model.pth")
        torch.save(self.model.state_dict(), save_path)

    def _log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def plot_metrics(self):
        epochs = range(1, self.args.max_epoch + 1)

        # Loss
        plt.figure()
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.test_losses, label="Test Loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_path, "loss_curve.png"))
        plt.close()

        # Accuracy
        plt.figure()
        plt.plot(epochs, self.train_accuracies, label="Train Acc")
        plt.plot(epochs, self.test_accuracies, label="Test Acc")
        plt.legend()
        plt.savefig(os.path.join(self.output_path, "accuracy_curve.png"))
        plt.close()


# =====================================================
# Args
# =====================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', default='./result', type=str)
    parser.add_argument('--data_root', default='/home/chenjiayi/workspace/willm/wifi_data', type=str)
    parser.add_argument('--dataset', default='RFNet', type=str)

    parser.add_argument('--model', default='ResNet101')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cpu', action='store_true')

    return parser.parse_args()


# =====================================================
# Main
# =====================================================

def main():
    mp.set_start_method("spawn", force=True)  # 防止底层 fork 崩溃

    args = get_args()
    set_seed(args.seed)

    print("\nRunning Experiment:")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}\n")

    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()