import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd

def summarize_data(config):

    labels_path = config["test_labels_path"]
    metadata_file = osp.join(labels_path, 'metadata_dataset_labels.csv')

    results_path = config["results_path"]
    dataset_report_file = osp.join(results_path, "test_dataset_report.jpg")

    metadata_df = pd.read_csv(metadata_file)
    metadata_exist_df = metadata_df.loc[metadata_df["exist"] == "Yes"]

    px = 1 / plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(1, 4, figsize=(1700 * px, 350 * px))

    def my_fmt2(x):
        return '{:.2f}%\n({:.0f})'.format(x, total2 * x / 100)
    def my_fmt(x):
        return '{:.2f}%\n({:.0f})'.format(x, total * x / 100)

    total2 = len(metadata_df)
    total = len(metadata_exist_df)
    axes[3].pie(metadata_exist_df.Shadow.value_counts(), autopct=my_fmt, colors=['orange', 'skyblue'], labels=['No', 'Yes'])
    axes[3].set_title("Images composition by existence of shadow")
    axes[2].pie(metadata_df.exist.value_counts(), autopct=my_fmt2, colors=['orange', 'skyblue'],labels=['Visible', 'Not-visible'])
    axes[2].set_title("Images composition by court visibility")
    axes[0].hist(metadata_exist_df["Brightness"])
    axes[0].grid(axis="y")
    axes[0].set_title("Composition of images by brightness level")
    axes[1].pie(metadata_exist_df.type.value_counts(), autopct=my_fmt, colors=['orange', 'skyblue'],
                   labels=['annexe', 'principale'])
    axes[1].set_title("Composition of images by court type")
    fig.tight_layout()

    plt.savefig(dataset_report_file)


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    summarize_data(config)
