import pandas as pd
import os
import os.path as osp
import datasets
import models
import exploitations
import experiment_definition as ed
import traceback
import random
import torch
import gc
import warnings
import sys
import mailer
import time
import dill
import shutil
import utils
warnings.filterwarnings("ignore")


def get_dataset(config):
    class_name = config["dataset_name"].capitalize() + "Dataset"
    class_ = getattr(datasets, class_name)
    class_args = dict({key: config[key] for key in ed.dataset_keys})
    return class_(**class_args)


def get_model(config, dataset):
    class_name = config["model_name"].capitalize() + "Model"
    class_ = getattr(models, class_name)
    class_args = dict({key: config[key] for key in ed.model_keys})
    return class_(dataset, **class_args)


def get_exploitation(config, model, dataset):
    class_name = config["exploitation_name"].capitalize() + "Exploitation"
    class_ = getattr(exploitations, class_name)
    class_args = dict({key: config[key] for key in ed.exploitation_keys})
    return class_(model, dataset, **class_args)


def run_training_experiment(config):
    dataset = get_dataset(config)
    dataset.prepare()

    model = get_model(config, dataset)
    model.prepare()
    model.train()
    model.test()
    model.unload()

    with open(osp.join(config["experiment_path"], "dumps", "dataset.pickle"), 'wb') as dataset_file:
        dill.dump(dataset, dataset_file)
    with open(osp.join(config["experiment_path"], "dumps", "model.pickle"), 'wb') as model_file:
        dill.dump(model, model_file)

    del dataset, model
    gc.collect()
    torch.cuda.empty_cache()


def run_results_experiment(config):
    with open(osp.join(config["experiment_path"], "dumps", "dataset.pickle"), 'rb') as dataset_file:
        dataset = dill.load(dataset_file)
    with open(osp.join(config["experiment_path"], "dumps", "model.pickle"), 'rb') as model_file:
        model = dill.load(model_file)

    exploitation = get_exploitation(config, model, dataset)
    exploitation.prepare_exploitation()
    exploitation.generate_outputs()
    exploitation.get_results(config)

    del dataset, model, exploitation
    gc.collect()
    torch.cuda.empty_cache()
    return config


def setup_experiment(config, onlyResults,fineTune):
    experiment_path = ed.get_experiment_path(config)
    if onlyResults:
        if not osp.exists(experiment_path):
            onlyResults = False

    if not onlyResults and not fineTune:
        if osp.exists(experiment_path):
            os.rename(experiment_path, experiment_path + "_old" + str(random.randint(0, 1000)))
        os.makedirs(experiment_path)
        os.makedirs(osp.join(experiment_path, "logs"))
        os.makedirs(osp.join(experiment_path, "dumps"))
        os.makedirs(osp.join(experiment_path, "checkpoints"))
    config["experiment_path"] = experiment_path


def run_experiment_by_index(config, df, index, mail=True, onlyResults=False, fineTune=False):
    experiment_start_time = time.time()
    for key in ed.columns_experiment:
        config[key] = df.iloc[index][key]
    config["shuffle"] = bool(config["shuffle"])
    config["augmentation_spatial"] = bool(config["augmentation_spatial"])
    config["augmentation_colour"] = bool(config["augmentation_colour"])
    config["keep"] = bool(config["keep"])
    config["freeze_encoder"] = bool(config["freeze_encoder"])
    config["fineTune"]=fineTune

    setup_experiment(config, onlyResults,fineTune)
    experiment_path = config["experiment_path"]

    try:
        if not onlyResults:
            run_training_experiment(config)
        time.sleep(0.5)
        result_config = run_results_experiment(dict(config))

    except Exception as e:
        error_msg = type(e).__name__ + "\n" + traceback.format_exc()
        print(error_msg)
        config["error"] = error_msg
        os.rename(experiment_path, experiment_path + "_error" + str(random.randint(0, 1000)))
    except KeyboardInterrupt as k:
        error_msg = type(k).__name__
        config["error"] = error_msg
        os.rename(experiment_path, experiment_path + "_error" + str(random.randint(0, 1000)))
        raise KeyboardInterrupt
    else:
        print(f"Experiment {result_config['experiment_id']} passed without errors")
        result_config["completed"] = True
        result_config["error"] = ""
        config = result_config

    print("\n" + "-" * 50 + "\n")
    config['experiment_elapsed_time'] = time.time() - experiment_start_time

    for key in ed.columns_experiment:
        df.at[index, key] = config[key]

    if mail:
        mailer.send_mail(config)
        print("Mail with results sent")

    if not config["keep"]:
        shutil.rmtree(osp.join(experiment_path, "checkpoints"))
        shutil.rmtree(osp.join(experiment_path, "dumps"))
        shutil.rmtree(osp.join(experiment_path, "Temp_dataset"))
        shutil.rmtree(osp.join(experiment_path, "Samples", "train"))
        shutil.rmtree(osp.join(experiment_path, "Samples", "valid"))

    return df


def assert_experiment_list(df):
    assert (df['experiment_id'].nunique() == len(df['experiment_id']))
    assert (df['dataset_name'].isin(ed.dataset_variants).all())
    assert (df['split_type'].isin(ed.split_variants).all())
    assert (df['model_name'].isin(ed.model_variants).all())
    assert (df['encoder'].isin(ed.encoders).all())
    assert (df['encoder_weights'].isin(ed.weights_type).all())
    assert (df['activation'].isin(ed.activations).all())
    assert (df['optimizer'].isin(ed.optimizers).all())
    assert (df['augmentation_colour_format'].isin(ed.colour_formats).all())
    #assert (df['loss_function'].isin(config["loss_functions"]).all())


def experiments_preprocessing(df):
    df.astype(ed.columns_experiment)
    return df


def experiments_pipeline(config, id=None, mail=True, onlyResults=False, fineTune=False):
    experiment_list_file = osp.join(config["experiments_path"], "experiments_list.csv")
    experiments_df = pd.read_csv(experiment_list_file)
    experiments_df = experiments_preprocessing(experiments_df)
    assert_experiment_list(experiments_df)

    if id is not None:
        index = experiments_df.loc[experiments_df['experiment_id'] == id].index.values.astype(int)[0]
        experiments_df = run_experiment_by_index(config, experiments_df, index, mail=mail, onlyResults=onlyResults, fineTune=fineTune)
        experiments_df.to_csv(experiment_list_file, index=False)
    else:
        for index in experiments_df.index:
            if not experiments_df['completed'][index]:
                experiments_df = run_experiment_by_index(config, experiments_df, index, mail=mail,
                                                         onlyResults=onlyResults)
                experiments_df.to_csv(experiment_list_file, index=False)
                shutil.copy(experiment_list_file, osp.join(config["experiments_path"], "experiments_list_backup.csv"))


if __name__ == "__main__":
    from config import load_config
    import logging

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s: %(message)s')

    config = load_config()
    # st=time.time()
    # while utils.get_gpu_memory() > 5000:
    #     time.sleep(300)
    #     print(f"{round((time.time()-st)/60)} minutes passed")
    # print("Dobby is free")

    experiments_pipeline(config, id=None, mail=True, onlyResults=False, fineTune=False)

