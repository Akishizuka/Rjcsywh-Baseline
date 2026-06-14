import csv
import datetime
import numpy as np
import tensorflow as tf

from donut import complete_timestamp, standardize_kpi, Donut, DonutTrainer, DonutPredictor
from tfsnippet.modules import Sequential
from tensorflow import keras as K


def parse_timestamp(ts_str):
    if ts_str.endswith('+00:00') or ts_str.endswith('-00:00'):
        ts_str = ts_str[:-3] + ts_str[-2:]

    for fmt in ('%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S'):
        try:
            dt = datetime.datetime.strptime(ts_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError('Unsupported timestamp format: {}'.format(ts_str))


def load_adservice_p99(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        ts = []
        values = []
        labels = []
        for row in reader:
            ts.append(parse_timestamp(row['timestamp']))
            values.append(float(row['adservice&grpc_latency_p99']))
            labels.append(
                1 if row['target_service'] == 'adservice' and row['fault_type'] != 'normal' else 0
            )

    return (
        np.array(ts, dtype=np.int64),
        np.array(values, dtype=np.float32),
        np.array(labels, dtype=np.int32),
    )


def evaluate_thresholds(labels, scores):
    """
    纯逐点（Point-wise）评估函数 —— 完全禁用 PA 策略
    """
    thresholds = np.unique(
        np.concatenate([
            np.linspace(scores.min(), scores.max(), num=500), # 提高搜索密度
            np.array([0.0, 0.5, 1.0], dtype=np.float32),
        ])
    )
    
    best = None
    for thr in thresholds:
        pred = (scores > thr).astype(np.int32)
        
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        if best is None or f1 > best[0]:
            best = (f1, thr, precision, recall, tp, fp, fn)
            
    return best


def main():
    csv_path = 'sample_data/final_dataset_for_algorithm.csv'
    timestamp, values, labels = load_adservice_p99(csv_path)

    timestamp, missing, (values, labels) = complete_timestamp(
        timestamp, (values, labels))

    test_portion = 0.3
    test_n = int(len(values) * test_portion)
    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    train_missing, test_missing = missing[:-test_n], missing[-test_n:]

    train_values, mean, std = standardize_kpi(
        train_values, excludes=np.logical_or(train_labels, train_missing))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

    # 构建 Donut 模型
    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001), activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001), activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001), activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001), activation=tf.nn.relu),
            ]),
            x_dims=60,   # 💡 优化1：缩短窗口大小到 60（10分钟），提高模型对突发异常反应的灵敏度
            z_dims=10,   
        )

    trainer = DonutTrainer(
        model=model,
        model_vs=model_vs,
        max_epoch=30,
        batch_size=64,
        valid_step_freq=100,
    )
    
    predictor = DonutPredictor(model, n_z=10, batch_size=32, last_point_only=True)

    with tf.Session().as_default():
        trainer.fit(train_values, train_labels, train_missing, mean, std)
        score = predictor.get_score(test_values, test_missing)

    test_labels = np.array(test_labels, dtype=np.int32)
    
    # 💡 优化2：异常得分平滑处理
    raw_anomaly_scores = -score
    # 使用 3 阶移动平均平滑，补齐边缘处偶发性的重构成功点，强力拉升 Recall
    smooth_window = 3
    anomaly_scores = np.convolve(raw_anomaly_scores, np.ones(smooth_window)/smooth_window, mode='same')

    aligned_test_labels = test_labels[model.x_dims - 1:]
    if anomaly_scores.shape[0] != aligned_test_labels.shape[0]:
        raise ValueError(
            'Score length does not match aligned label length: {} vs {}'.format(
                anomaly_scores.shape[0], aligned_test_labels.shape[0]))

    # 寻找最佳阈值并评估（纯逐点结果）
    best = evaluate_thresholds(aligned_test_labels, anomaly_scores)
    if best is not None:
        best_f1, best_thr, best_p, best_r, best_tp, best_fp, best_fn = best
        print('=== 优化后的评测结果 (纯逐点 Point-wise，无 PA) ===')
        print('Best Threshold :', best_thr)
        print('Precision      :', round(best_p, 4))
        print('Recall         :', round(best_r, 4))
        print('F1-Score       :', round(best_f1, 4))
        print('Detail         : tp={}, fp={}, fn={}'.format(best_tp, best_fp, best_fn))

    out_path = 'adservice_p99_scores.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['index', 'reconstruction_prob', 'anomaly_score'])
        for i, s in enumerate(score):
            writer.writerow([i, float(s), float(-s)])

    print('\n[INFO] Saved scores to', out_path)


if __name__ == '__main__':
    main()