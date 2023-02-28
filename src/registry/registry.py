import io

import matplotlib.pyplot as plt
import polars as pl

class Registry:
    def __init__(self, eval_root:str, data_root:str) -> None:
        self.eval_root = eval_root
        self.data_root = data_root

    def plot_training_metrics(self, run_id:str, episodes:int, total_rewards:list, max_rewards:list, algorithm:str) -> None:
        x_list = list(range(1, episodes + 1))
        y_lists = [total_rewards, max_rewards]
        plot_labels = ["Reward", "Max reward"]
        x_label = "Episode"
        y_label = "Reward"
        title = f"Training - Rewards ({algorithm})"
        filename = f"plot-training-{algorithm}-rewards.png"
        self.save_plot(run_id, x_list, y_lists, plot_labels, x_label, y_label, title, filename)

    def save_plot(self, run_id, x_list, y_lists, plot_labels, x_label, y_label, title, filename):
        plt.switch_backend('agg')
        for i in range(0, len(y_lists)):
            plt.plot(x_list, y_lists[i], label=plot_labels[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer)
        plt.close()

        self.write_bytes(run_id, buffer, filename)

    def save_bar(self, run_id, x_list, y_list, x_label, y_label, title, filename):
        plt.switch_backend('agg')
        plt.bar(x_list, y_list)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        buffer = io.BytesIO()
        plt.savefig(buffer)
        plt.close()

        self.write_bytes(run_id, buffer, filename)

    def save_model(self, run_id:str, filename:str, buffer:io.BytesIO) -> None:
        self.write_bytes(run_id, buffer, filename)

    def load_model(self, run_id:str, filename:str) -> io.BytesIO:
        return self.read_bytes(run_id, filename)

    def read_training_data(self, path:str, filename:str) -> pl.DataFrame:
        raise Exception(f"Function not implemented in {__name__}")

    def write_bytes(self, path:str, buffer:io.BytesIO, filename:str, root:str=None) -> None:
        raise Exception(f"Function not implemented in {__name__}")

    def read_bytes(self, path:str, filename:str, root:str=None) -> io.BytesIO:
        raise Exception(f"Function not implemented in {__name__}")
