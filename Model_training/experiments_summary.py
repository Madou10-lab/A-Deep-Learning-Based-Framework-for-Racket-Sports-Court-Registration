import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
pd.options.mode.chained_assignment = None

experiments_groups = [
    ("augmentation variation", 3, [3, 4, 6, 7]),
    ("dataset ablation", 32, [32, 34, 35, 36]),
    ("efficientnet effect", 32, [32, 50, 51]),
    ("ablation split", 65, [65,66,67,68,34,35,36]),
    ("Pybad variations", 65, [65,70,72,75]),
    ("Pybad alphas", 65, [65,76,77,80,81])
]


def summarize_miou(df, baseline,group_name, path):
    df_sorted = df.sort_values(by='test_miou', ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)

    x_row = df_sorted[df_sorted['experiment_id'] == baseline]
    df_sorted.drop(x_row.index, inplace=True)
    df_sorted = pd.concat([x_row, df_sorted])
    df_sorted.reset_index(drop=True, inplace=True)
    df = df_sorted[['experiment_id','experiment_name', 'train_miou', 'valid_miou', 'test_miou']]
    df.loc[:, 'experiment_id_name'] = df['experiment_id'].astype(str) + '_' + df['experiment_name']
    plt.figure(figsize=(12, 6))

    bar_width = 0.1
    index = range(len(df))

    plt.bar([i - bar_width for i in index], df['train_miou'], width=bar_width, color='b', align='edge',
            label='train_miou')
    plt.bar(index, df['valid_miou'], width=bar_width, color='g', align='edge', label='valid_miou')
    plt.bar([i + bar_width for i in index], df['test_miou'], width=bar_width, color='r', align='edge',
            label='test_miou')

    plt.xticks(index,df['experiment_id_name'])
    plt.ylim([df['test_miou'].min() - 0.05, 1])

    # Set labels and title
    plt.xlabel('Experiment_name')
    plt.ylabel('mIoU')
    plt.grid(axis="y")
    plt.title(f'Comparison of mIoU for {group_name}')

    plt.legend()
    plt.savefig(path)
    print("Miou summaries saved")
    #plt.show()

def summarize_inference(df, baseline,group_name, path):
    df_sorted = df.sort_values(by='test_miou', ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)

    x_row = df_sorted[df_sorted['experiment_id'] == baseline]
    df_sorted.drop(x_row.index, inplace=True)
    df_sorted = pd.concat([x_row, df_sorted])
    df_sorted.reset_index(drop=True, inplace=True)
    df = df_sorted[['experiment_id','experiment_name', 'test_miou','inference_time','inference_gpu_usage','video_fps','video_elapsed_time']]
    df.loc[:, 'experiment_id_name'] = df['experiment_id'].astype(str) + '_' + df['experiment_name']

    px = 1 / plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(2, 2, figsize=(1200 * 2 * px, 800 * 2 * px))

    axes[0][0].set_xlabel('Experiment_name', fontsize=14)
    axes[0][1].set_xlabel('Experiment_name', fontsize=14)
    axes[1][0].set_xlabel('Experiment_name', fontsize=14)
    axes[1][1].set_xlabel('Experiment_name', fontsize=14)

    axes[0][0].set_ylabel("Seconds", fontsize=14)
    axes[0][0].set_ylabel("Megabytes", fontsize=14)
    axes[1][0].set_ylabel("FPS", fontsize=14)
    axes[1][0].set_ylabel("Seconds", fontsize=14)

    axes[0][0].grid(axis="y")
    axes[0][1].grid(axis="y")
    axes[1][0].grid(axis="y")
    axes[1][1].grid(axis="y")

    axes[0][0].set_title('Comparison of inference time', fontsize=14)
    axes[0][1].set_title('Comparison of gpu memory in inference', fontsize=14)
    axes[1][0].set_title('Comparison of average video fps', fontsize=14)
    axes[1][1].set_title('Comparison of elapsed time for video', fontsize=14)

    bar_width = 0.2
    index = range(len(df))

    axes[0][0].bar(index, df['inference_time'], width=bar_width, color='b', align='edge')
    axes[0][1].bar(index, df['inference_gpu_usage'], width=bar_width, color='r', align='edge')
    axes[1][0].bar(index, df['video_fps'], width=bar_width, color='g', align='edge')
    axes[1][1].bar(index, df['video_elapsed_time'], width=bar_width, color='orange', align='edge')

    axes[0][0].set_xticks(index,df['experiment_id_name'])
    axes[0][1].set_xticks(index,df['experiment_id_name'])
    axes[1][0].set_xticks(index,df['experiment_id_name'])
    axes[1][1].set_xticks(index,df['experiment_id_name'])

    plt.savefig(path)
    #plt.show()
    print("Inference summaries saved")

def summarize_train_valid(df, baseline, experiments_path, path):
    df_sorted = df.sort_values(by='test_miou', ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)

    x_row = df_sorted[df_sorted['experiment_id'] == baseline]
    df_sorted.drop(x_row.index, inplace=True)
    df_sorted = pd.concat([x_row, df_sorted])
    df_sorted.reset_index(drop=True, inplace=True)

    px = 1 / plt.rcParams['figure.dpi']
    fig, axes = plt.subplots(2, 2, figsize=(1200 * 2 * px, 800 * 2 * px))

    axes[0][0].set_xlabel('Epochs', fontsize=14)
    axes[0][1].set_xlabel('Epochs', fontsize=14)
    axes[0][0].set_ylabel("Train mIoU", fontsize=14)
    axes[0][0].set_ylabel("Valid mIoU", fontsize=14)
    axes[0][0].grid()
    axes[0][1].grid()
    axes[0][0].set_title('Comparison of train mIoU Plots', fontsize=14)
    axes[0][1].set_title('Comparison of valid mIoU Plots', fontsize=14)

    axes[1][0].set_xlabel('Epochs', fontsize=14)
    axes[1][1].set_xlabel('Epochs', fontsize=14)
    axes[1][0].set_ylabel("Train mIoU", fontsize=14)
    axes[1][0].set_ylabel("Valid mIoU", fontsize=14)
    axes[1][0].grid()
    axes[1][1].grid()
    axes[1][0].set_title('Comparison of zoomed train mIoU Plots', fontsize=14)
    axes[1][1].set_title('Comparison of zoomed valid mIoU Plots', fontsize=14)
    axes[1][0].set_ylim([0.95,1])
    axes[1][1].set_ylim([0.95,1])

    for index in df_sorted.index:
        dirname = f"{df_sorted['experiment_id'][index]}_{df_sorted['model_name'][index]}_{df_sorted['dataset_name'][index]}_{df_sorted['experiment_name'][index]}"
        experiment_path = osp.join(experiments_path, dirname)
        train_valid_df=pd.read_csv(osp.join(experiment_path,"training_validation_data.csv"))
        train_valid_df_zoomed = train_valid_df.iloc[10:]

        label=str(df_sorted["experiment_id"][index]) + '_' +df_sorted["experiment_name"][index]

        axes[0][0].plot(train_valid_df.index.tolist(), train_valid_df["train_iou_score"].tolist(), lw=2, label=label)
        axes[0][1].plot(train_valid_df.index.tolist(), train_valid_df["valid_iou_score"].tolist(), lw=2, label=label)
        axes[1][0].plot(train_valid_df_zoomed.index.tolist(), train_valid_df_zoomed["train_iou_score"].tolist(), lw=2,
                        label=label)
        axes[1][1].plot(train_valid_df_zoomed.index.tolist(), train_valid_df_zoomed["valid_iou_score"].tolist(), lw=2,
                        label=label)

    axes[0][0].legend(loc='best', fontsize=12)
    axes[0][1].legend(loc='best', fontsize=12)
    axes[1][0].legend(loc='best', fontsize=12)
    axes[1][1].legend(loc='best', fontsize=12)

    plt.savefig(path)
    #plt.show()
    print("Training graph summaries saved")


#TODO: add tables instead of bar plot
def summarize_experiments(config):
    experiment_list_file = osp.join(config["experiments_path"], "experiments_list.csv")
    experiments_df = pd.read_csv(experiment_list_file)
    results_path = config["results_path"]
    i=1
    for group_name, baseline, exp_group in experiments_groups:
        df = experiments_df[experiments_df['experiment_id'].isin(exp_group)]
        summarize_inference(df, baseline, group_name, osp.join(results_path, f"{i}_{group_name.replace(' ', '_')}_inference.jpg"))
        summarize_miou(df, baseline, group_name, osp.join(results_path, f"{i}_{group_name.replace(' ', '_')}_miou.jpg"))
        summarize_train_valid(df, baseline, config["experiments_path"], osp.join(results_path, f"{i}_{group_name.replace(' ', '_')}_train.jpg"))
        i+=1


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    summarize_experiments(config)
