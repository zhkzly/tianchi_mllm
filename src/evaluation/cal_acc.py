
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json


def calculate_metrics(y_true, y_pred):
    """
    计算多标签分类任务的指标: 加权F1, 总体acc, precision, recall。

    参数:
    y_true (List[List[int]]): 真实标签
    y_pred (List[List[int]]): 预测标签

    返回:
    dict: 各指标的分数
    """

    # 计算指标
    metrics = {
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
    }

    return metrics


def get_label_pred(test_file, pred_file):
    """获取测试集标签以及预测结果

    Args:
        test_file (_type_): 带ground_truth标签的测试集文件
        pred_file (_type_): 对应的预测结果文件
    """
    test_data = json.load(open(test_file, "r"))
    labels = [item["output"] for item in test_data]

    pred_data = [json.loads(line) for line in open(pred_file, "r")]
    preds = [item["predict"] for item in pred_data]

    return labels, preds


def cal_acc(test_file, pred_file):
    labels, preds = get_label_pred(test_file, pred_file)
    metrics = calculate_metrics(y_true=labels, y_pred=preds)
    return metrics


if __name__ == "__main__":
    test_file = "data/demo_test.json"
    pred_file = "data/demo_pred.jsonl"
    metrics = cal_acc(test_file, pred_file)

    print(metrics)
