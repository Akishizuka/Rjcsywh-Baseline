import torch

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from dataloader import *
from models.CutAddPaste.trainer.trainer import Trainer
from models.CutAddPaste.network.model import base_Model
from models.reasonable_metric import reasonable_accumulator
from ts_datasets.ts_datasets.anomaly import NAB, IOpsCompetition, SMAP, SMD, UCR, MSL
from tqdm import tqdm
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from utils import print_object
import matplotlib.pyplot as plt


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--visualization', default=False, type=bool,
                    help='Visualize')
parser.add_argument('--seed', default=4, type=int,
                    help='seed value')
parser.add_argument('--selected_dataset', default='IOpsCompetition', type=str,
                    help='Dataset of choice: IOpsCompetition, UCR, NAB, SWaT, WADI, SMD, SMAP, MSL')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()


def _load_local_numpy_dataset(dataset_name, base_dir):
    dataset_dirs = [dataset_name.lower()]
    if dataset_name == "IOpsCompetition":
        dataset_dirs = ["aiops", "iops_competition"]

    dataset_dir = None
    for d in dataset_dirs:
        candidate = os.path.join(base_dir, d)
        if os.path.isdir(candidate):
            dataset_dir = candidate
            break
    if dataset_dir is None:
        return None

    files = os.listdir(dataset_dir)
    train_file = next((f for f in files if "train" in f.lower() and f.lower().endswith(".npy")), None)
    test_file = next(
        (
            f
            for f in files
            if "test" in f.lower() and "label" not in f.lower() and f.lower().endswith(".npy")
        ),
        None,
    )
    label_file = next((f for f in files if "test" in f.lower() and "label" in f.lower() and f.lower().endswith(".npy")), None)

    if not (train_file and test_file and label_file):
        return None

    def _load_raw(path):
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
        return raw

    def _is_single_metric_source(raw):
        if isinstance(raw, dict):
            return True
        arr = np.asarray(raw)
        return hasattr(arr, "dtype") and arr.dtype.names is not None

    def _extract_series(raw, preferred_key="value"):
        if isinstance(raw, dict):
            if preferred_key in raw:
                arr = np.asarray(raw[preferred_key])
            else:
                # 回退到第一个可用的一维数值键
                candidates = [k for k in raw.keys() if k.lower() not in ["timestamp", "kpi id"]]
                if len(candidates) == 0:
                    raise ValueError("No usable key found in local npy dict.")
                arr = np.asarray(raw[candidates[0]])
        else:
            arr = np.asarray(raw)
            # np structured array (e.g., fields: timestamp/value/label/KPI ID)
            if hasattr(arr, "dtype") and arr.dtype.names is not None:
                names = list(arr.dtype.names)
                if preferred_key in names:
                    arr = np.asarray(arr[preferred_key])
                else:
                    candidates = [k for k in names if k.lower() not in ["timestamp", "kpi id"]]
                    if len(candidates) == 0:
                        raise ValueError("No usable field found in local structured array.")
                    arr = np.asarray(arr[candidates[0]])
        return arr

    train_raw = _load_raw(os.path.join(dataset_dir, train_file))
    test_raw = _load_raw(os.path.join(dataset_dir, test_file))
    label_raw = _load_raw(os.path.join(dataset_dir, label_file))

    single_metric = (
        _is_single_metric_source(train_raw)
        and _is_single_metric_source(test_raw)
        and _is_single_metric_source(label_raw)
    )

    train_data = _extract_series(train_raw, preferred_key="value")
    test_data = _extract_series(test_raw, preferred_key="value")
    test_labels = _extract_series(label_raw, preferred_key="label")

    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)
    elif train_data.ndim > 1 and single_metric:
        # SMAP/MSL/aiops 这类单指标数据，只保留第 1 个通道
        train_data = train_data.reshape(train_data.shape[0], -1)[:, :1]
    if test_data.ndim == 1:
        test_data = test_data.reshape(-1, 1)
    elif test_data.ndim > 1 and single_metric:
        test_data = test_data.reshape(test_data.shape[0], -1)[:, :1]
    if test_labels.ndim > 1:
        test_labels = test_labels.reshape(-1)

    # Unsupervised setting: training labels are all normal points.
    train_labels = np.zeros(train_data.shape[0], dtype=int)
    test_labels = (test_labels > 0).astype(int)
    return train_data, test_data, train_labels, test_labels

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'CutAddPaste'
run_description = args.run_description
selected_dataset = args.selected_dataset
visualization = args.visualization

exec(f'from conf.cut_add_paste.{data_type}_Configs import Config as Configs')
configs = Configs()
weight_decay = configs.weight

# ##### fix random seeds for reproducibility ########
SEED = args.seed

print(f'Dataset: {data_type}')
print(f'Method:  {method}')
print(f'Random Seed:  {SEED}')

# Load datasets
if selected_dataset == 'SWaT':
    model_num = 1
elif selected_dataset == 'WADI':
    model_num = 1
else:
    # 如果存在 data/<dataset>/ 的本地预处理 npy，则优先使用（避免联网下载）
    local_numpy_data = _load_local_numpy_dataset(selected_dataset, os.path.join(args.home_path, "data"))

    if local_numpy_data is not None:
        dt = None
        model_num = 1
    else:
        if selected_dataset == 'NAB':
            dt = NAB()
        elif selected_dataset == 'IOpsCompetition':
            dt = IOpsCompetition()
        elif selected_dataset == 'SMAP':
            dt = SMAP()
        elif selected_dataset == 'MSL':
            dt = MSL()
        elif selected_dataset == 'SMD':
            dt = SMD()
        elif selected_dataset == 'UCR':
            dt = UCR()
        else:
            dt = SMD()
        model_num = len(dt)
    # model_num = 1

# Aggregate statistics from full dataset
all_test_rpa_score, all_test_pa_score, all_test_pw_score = [], [], []
all_anomaly_num, all_test_scores_reasonable = [], []
all_test_aff_score, all_test_aff_precision, all_test_aff_recall = [], [], []
all_test_roc_auc = []
detect_list = np.zeros(model_num)
for idx in tqdm(range(model_num)):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)

    if selected_dataset == 'SWaT':
        train_data, test_data, train_labels, test_labels = swat()
    elif selected_dataset == 'WADI':
        train_data, test_data, train_labels, test_labels = wadi()
    elif local_numpy_data is not None:
        train_data, test_data, train_labels, test_labels = local_numpy_data
    else:
        time_series, meta_data = dt[idx]
        train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta_data)

    print('>' * 32, len(train_data))
    print('>' * 32, len(test_data))

    # Load Model
    model = base_Model(configs, device).to(device)
    print("Data loaded ...")
    train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator(train_data, test_data, train_labels,
                                                                         test_labels, SEED, configs)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=weight_decay)

    # Trainer
    (
        test_score_origin,
        test_aff,
        test_rpa_score,
        test_pa_score,
        test_pw_score,
        score_reasonable,
        predict,
        test_roc_auc,
    ) = Trainer(
        model,
        model_optimizer,
        train_dl,
        val_dl,
        test_dl,
        device,
        configs,
        idx,
    )

    all_anomaly_num.append(test_anomaly_window_num)
    all_test_scores_reasonable.append(score_reasonable)
    all_test_aff_precision.append(test_aff["precision"])
    all_test_aff_recall.append(test_aff["recall"])
    all_test_aff_score.append(test_aff)
    all_test_rpa_score.append(test_rpa_score)
    all_test_pa_score.append(test_pa_score)
    all_test_pw_score.append(test_pw_score)
    all_test_roc_auc.append(test_roc_auc)

    # visualization
    if visualization:
        print('*'*32)
        fig = plt.figure(facecolor="w", figsize=(10, 6))
        ax = fig.add_subplot(111)
        # train_data_plot = time_series[meta_data.trainval]
        # train_labels_plot = meta_data.anomaly[meta_data.trainval]
        test_data_plot = time_series[~meta_data.trainval]
        test_labels_plot = meta_data.anomaly[~meta_data.trainval]
        # plot time-series value
        t_data, y_data = test_data_plot.index, test_data_plot.values
        t = np.arange(0, len(y_data), 1)
        g = len(y_data.shape)
        if g > 1:
            y_data = y_data[:, 0]
        ax.plot(t, y_data, linewidth=1)
        ax.set_ylabel('value', fontsize=10)

        # plot ground-truth anomaly
        t_label, y_label = test_labels_plot.index, test_labels_plot.values
        splits = np.where(y_label[1:] != y_label[:-1])[0] + 1
        splits = np.concatenate(([0], splits, [len(y_label) - 1]))
        for k in range(len(splits) - 1):
            if y_label[splits[k]]:  # If splits[k] is anomalous
                ax.axvspan(t[splits[k]], t[splits[k + 1]], color="#e07070", alpha=0.5)
        # plot predict anomaly score
        predict = np.tile(predict.reshape(-1, 1), configs.time_step).flatten()
        t_pred = np.arange(0, len(predict), 1)
        ax2 = ax.twinx()
        ax2.set_ylabel('anomaly', fontsize=10)
        ax2.plot(t_pred, predict, linewidth=1, color='r')
        time_series_name = test_data_plot.columns[0]
        plt.title(time_series_name + '_' + str(idx))
        plt.show()

        fig_origin = plt.figure(facecolor="w", figsize=(20, 12))
        ax_origin = fig_origin.add_subplot(111)
        test_score_origin = np.array(test_score_origin).reshape(-1, 1)
        test_score_origin = np.tile(test_score_origin, configs.time_step).flatten()
        ax_origin.plot(test_score_origin, linewidth=1)
        plt.tight_layout()
        plt.show()

        test_aff_f1 = 2 * (test_aff["precision"] * test_aff["recall"]) / (test_aff["precision"] + test_aff["recall"])
        detect_list[idx] = test_aff_f1

# visualization
if visualization:
    fig_all = plt.figure(facecolor="w", figsize=(20, 12))
    ax_all = fig_all.add_subplot(111)
    ax_all.plot(detect_list, linewidth=1)
    plt.show()
    np.savetxt("result/detect_list.csv", detect_list, delimiter=",")

all_anomaly_num = np.array(all_anomaly_num)
sum_anomaly_num = np.sum(all_anomaly_num)
all_test_aff_precision = np.array(all_test_aff_precision)
all_test_aff_precision = all_test_aff_precision * all_anomaly_num / sum_anomaly_num
test_aff_precision = np.nansum(all_test_aff_precision)
all_test_aff_recall = np.array(all_test_aff_recall)
all_test_aff_recall = all_test_aff_recall * all_anomaly_num / sum_anomaly_num
test_aff_recall = np.nansum(all_test_aff_recall)
if test_aff_precision + test_aff_recall == 0:
    test_aff_f1 = 0
else:
    test_aff_f1 = 2 * (test_aff_precision * test_aff_recall) / (test_aff_precision + test_aff_recall)

total_test_rpa_score = sum(all_test_rpa_score, ScoreAcc())
total_test_pa_score = sum(all_test_pa_score, ScoreAcc())
total_test_pw_score = sum(all_test_pw_score, ScoreAcc())
total_test_scores_reasonable = sum(all_test_scores_reasonable, reasonable_accumulator())
ucr_accuracy = total_test_scores_reasonable.get_all_metrics()

mean_test_roc_auc = float(np.nanmean(np.asarray(all_test_roc_auc, dtype=float))) if len(all_test_roc_auc) else float("nan")

print('>' * 32)
if configs.dataset == 'UCR':
    print("UCR metrics:\n",
          f"accuracy: {ucr_accuracy}\n")
if np.isfinite(mean_test_roc_auc):
    print(f"Test ROC-AUC (window-level, mean over series): {mean_test_roc_auc:.5f}")
else:
    print("Test ROC-AUC (window-level): undefined")
print("affiliation metrics:\n",
      f"Precision: {test_aff_precision:.5f}\n",
      f"Recall:    {test_aff_recall:.5f}\n"
      f"F1 Scores: {test_aff_f1:.5f}\n"
      "Revised-point-adjusted metrics:\n",
      f"Precision: {total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
      f"Recall:    {total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n"
      f"F1 score:  {total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n",
      "Point-adjusted metrics:\n",
      f"Precision: {total_test_pa_score.precision(ScoreType.PointAdjusted):.5f}\n",
      f"Recall:    {total_test_pa_score.recall(ScoreType.PointAdjusted):.5f}\n",
      f"F1 score:  {total_test_pa_score.f1(ScoreType.PointAdjusted):.5f}\n",
      "Point-wise metrics:\n",
      f"Precision: {total_test_pw_score.precision(ScoreType.Pointwise):.5f}\n",
      f"Recall:    {total_test_pw_score.recall(ScoreType.Pointwise):.5f}\n"
      f"F1 Scores:  {total_test_pw_score.f1(ScoreType.Pointwise):.5f}\n"
      "NAB Scores:\n",
      f"NAB Score (balanced):       {total_test_pa_score.nab_score():.5f}\n",
      f"NAB Score (high precision): {total_test_pa_score.nab_score(fp_weight=0.22):.5f}\n",
      f"NAB Score (high recall):    {total_test_pa_score.nab_score(fn_weight=2.0):.5f}\n"
      "seed:", SEED, "\n"
      "config setup:\n"
      )
str_conf = print_object(configs)
train_time = datetime.now()-start_time
print(f"Training time is : {train_time}")

path = "./results"
if not os.path.exists(path):
    os.makedirs(path)

summary = os.path.join("results", f"{method}_{selected_dataset}_summary.csv")
if os.path.exists(summary):
    df = pd.read_csv(summary, index_col=0)
else:
    df = pd.DataFrame()
model_name = method + f"{df.shape[1]}"

df.loc["Hyper-parameter", model_name] = str_conf
df.loc["seed", model_name] = SEED
df.loc["Train Time", model_name] = train_time
df.loc["UCR Accuracy", model_name] = round(ucr_accuracy["accuracy"], 5)
df.loc["Affiliation Precision", model_name] = round(test_aff_precision, 5)
df.loc["Affiliation Recall", model_name] = round(test_aff_recall, 5)
df.loc["Affiliation F1", model_name] = round(test_aff_f1, 5)
df.loc["RPA Precision", model_name] = round(total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA Recall", model_name] = round(total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA F1", model_name] = round(total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted), 5)
df.loc["PA Precision", model_name] = round(total_test_pa_score.precision(ScoreType.PointAdjusted), 5)
df.loc["PA Recall", model_name] = round(total_test_pa_score.recall(ScoreType.PointAdjusted), 5)
df.loc["PA F1", model_name] = round(total_test_pa_score.f1(ScoreType.PointAdjusted), 5)
df.loc["Point-wise Precision", model_name] = round(total_test_pw_score.precision(ScoreType.Pointwise), 5)
df.loc["Point-wise Recall", model_name] = round(total_test_pw_score.recall(ScoreType.Pointwise), 5)
df.loc["Point-wise F1", model_name] = round(total_test_pw_score.f1(ScoreType.Pointwise), 5)
if np.isfinite(mean_test_roc_auc):
    df.loc["Test ROC-AUC (window)", model_name] = round(mean_test_roc_auc, 5)
df.loc["NAB Score (balanced)", model_name] = round(total_test_rpa_score.nab_score(), 5)
df.loc["NAB Score (high precision)", model_name] = round(total_test_rpa_score.nab_score(fp_weight=0.22), 5)
df.loc["NAB Score (high recall)", model_name] = round(total_test_rpa_score.nab_score(fn_weight=2.0), 5)


df.to_csv(summary, index=True)
