#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/11/11 15:13:29
@Author  :   kaixinpangpangyu
@Version :   1.0
@Contact :   wangjinbo_0217@163.com
@Motto   :   Innovate Today
'''


import os
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from settings.config import settings


def project_root() -> str:
    # src/utils.py -> src -> 项目根目录
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_moirai2_local_path() -> str:
    root = project_root()
    local_dir = os.path.join(root, settings.models_dirname, settings.moirai2_local_dirname)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"未找到本地 Moirai2 模型目录：{local_dir}。请确保模型快照存在。"
        )
    # 校验必要文件是否存在
    cfg = os.path.join(local_dir, "config.json")
    st = os.path.join(local_dir, "model.safetensors")
    if not (os.path.isfile(cfg) and os.path.isfile(st)):
        raise FileNotFoundError(
            f"Moirai2 本地快照缺少必要文件：{local_dir}。应包含 config.json 与 model.safetensors。"
        )
    return local_dir


def load_csv_target(csv_path: str, target_column: str) -> np.ndarray:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到 CSV 文件：{csv_path}")
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于 CSV。")
    values = df[target_column].astype(float).to_numpy()
    return values


def load_csv_date_series(csv_path: str, date_column: str = "date") -> pd.Series:
    """读取 CSV 的日期列为时间戳序列。

    - 期望格式：YYYY-MM-DD hh:mm:ss（可被 pandas 解析）
    - 成功返回时间戳 `Series`；失败抛出异常，由调用方兜底。
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到 CSV 文件：{csv_path}")
    df = pd.read_csv(csv_path)
    if date_column not in df.columns:
        raise ValueError(f"日期列 '{date_column}' 不存在于 CSV。")
    ts = pd.to_datetime(df[date_column], errors="coerce")
    if ts.isna().any():
        raise ValueError("日期列存在无法解析的时间戳值。")
    return ts


def load_csv_target_and_covariates(csv_path: str, target_column: str, date_column: str = "date") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到 CSV 文件：{csv_path}")
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于 CSV。")
    target = df[target_column].astype(float).to_numpy()
    cols = [c for c in df.columns if c not in {target_column, date_column}]
    num_df = df[cols].select_dtypes(include=[np.number])
    cov_cols = list(num_df.columns)
    if len(cov_cols) == 0:
        return target, np.zeros((0, len(target)), dtype=float), []
    covs = num_df.astype(float).to_numpy().T
    return target, covs, cov_cols


def build_context_only_listdataset(
    target: np.ndarray,
    freq: str,
    used_ctx: int,
    csv_path: Optional[str] = None,
    date_column: str = "date",
    past_covs: Optional[np.ndarray] = None,
) -> ListDataset:
    """仅基于序列最后 `used_ctx` 步构造数据集，用于纯外推预测。

    - `used_ctx`：实际使用的上下文长度（已按序列长度裁剪）。
    - 返回包含一个条目的 `ListDataset`，其 `target` 为末尾 `used_ctx` 个点。
    """
    if used_ctx <= 0:
        raise ValueError("used_ctx 必须大于 0。")
    # 优先尝试从 CSV 读取真实起始时间戳；失败则降级到占位时间戳
    start_ts = pd.Timestamp("2000-01-01 00:00:00")
    if csv_path is not None:
        try:
            dates = load_csv_date_series(csv_path, date_column)
            if len(dates) != int(target.shape[0]):
                logger.warning(
                    "读取日期列成功，但长度不匹配：date_len={}，target_len={}；使用占位时间戳。",
                    len(dates),
                    int(target.shape[0]),
                )
            else:
                start_idx = int(target.shape[0]) - used_ctx
                start_ts = pd.Timestamp(dates.iloc[start_idx])
                logger.info("已使用 CSV 日期列作为起始时间戳：{}（索引={}）", start_ts, start_idx)
        except Exception as e:
            logger.warning("读取日期列失败，降级使用占位时间戳；原因：{}", e)
    ctx = target[-used_ctx:]
    entry = {"start": start_ts, "target": ctx}
    if past_covs is not None and past_covs.size > 0:
        entry["past_feat_dynamic_real"] = past_covs[:, -used_ctx:]
    return ListDataset([entry], freq=freq)


def make_sliding_context_and_labels(
    target: np.ndarray,
    context_length: int,
    prediction_length: int,
    step: int | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """基于整段序列生成滑动窗口的上下文与标签对。

    - 第一个窗口覆盖 `[0 : context_length + prediction_length)`；
    - 后续窗口每次右移 `step`（默认与 `prediction_length` 一致）；
    - 若最后一个窗口不足 `context_length + prediction_length`，则舍弃。

    返回：两个列表，`contexts` 与 `labels`，长度一致。
    """
    n = int(target.shape[0])
    if step is None:
        step = prediction_length
    if context_length <= 0 or prediction_length <= 0:
        raise ValueError("context_length 与 prediction_length 必须为正数。")
    contexts: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    window = context_length + prediction_length
    if n < window:
        return contexts, labels
    start = 0
    while start + window <= n:
        ctx = target[start : start + context_length]
        lab = target[start + context_length : start + window]
        contexts.append(ctx.astype(float))
        labels.append(lab.astype(float))
        start += step
    return contexts, labels


def build_listdataset_from_contexts(
    contexts: List[np.ndarray],
    freq: str,
    csv_path: Optional[str] = None,
    date_column: str = "date",
    step: Optional[int] = None,
    past_covs: Optional[np.ndarray] = None,
    context_length: Optional[int] = None,
) -> ListDataset:
    """将多个上下文片段封装为 GluonTS 的 `ListDataset`。

    - 每个条目的 `target` 为一个上下文序列。
    - 默认使用固定的起始时间戳占位；若提供 `csv_path` 且能读取日期列，则每个窗口使用其真实起始时间戳。
    """
    # 默认占位时间戳
    default_ts = pd.Timestamp("2000-01-01 00:00:00")

    # 若提供 csv_path 并且 step 有值，则尝试读取真实起始时间戳
    if csv_path is not None and step is not None:
        try:
            dates = load_csv_date_series(csv_path, date_column)
            entries = []
            for i, c in enumerate(contexts):
                start_idx = i * int(step)
                if start_idx >= len(dates):
                    logger.warning(
                        "窗口起始索引超出日期序列范围：start_idx={}，date_len={}；使用占位时间戳。",
                        start_idx,
                        len(dates),
                    )
                    e = {"start": default_ts, "target": c}
                    if past_covs is not None and past_covs.size > 0 and context_length is not None:
                        e["past_feat_dynamic_real"] = past_covs[:, start_idx : start_idx + int(context_length)]
                    entries.append(e)
                else:
                    st = pd.Timestamp(dates.iloc[start_idx])
                    e = {"start": st, "target": c}
                    if past_covs is not None and past_covs.size > 0 and context_length is not None:
                        e["past_feat_dynamic_real"] = past_covs[:, start_idx : start_idx + int(context_length)]
                    entries.append(e)
            logger.info("滑动窗口已使用 CSV 日期列作为起始时间戳，共 {} 个窗口。", len(entries))
            return ListDataset(entries, freq=freq)
        except Exception as e:
            logger.warning("读取日期列失败，滑动窗口降级使用占位时间戳；原因：{}", e)

    # 兜底：全部使用占位时间戳
    entries = []
    for i, c in enumerate(contexts):
        e = {"start": default_ts, "target": c}
        if past_covs is not None and past_covs.size > 0 and context_length is not None and step is not None:
            start_idx = i * int(step)
            e["past_feat_dynamic_real"] = past_covs[:, start_idx : start_idx + int(context_length)]
        entries.append(e)
    return ListDataset(entries, freq=freq)


def create_tail_test_instances(full_ds: ListDataset, prediction_length: int):
    # 在序列尾部偏移处切分以生成单次预测窗口
    test_input, test_template = split(full_ds, offset=-prediction_length)  # N - prediction_length, prediction_length 
    test_data = test_template.generate_instances(prediction_length)  # test_data.input(context_length) ：预测输入迭代器; test_data.label(prediction_length) ：真实标签迭代器
    return test_data


def compute_metadata(target: np.ndarray, train_ratio: float, prediction_length: int, feature: str = "S", past_feat_dim: int = 0, future_feat_dim: int = 0) -> Dict:
    total_len = int(target.shape[0])
    train_len = int(total_len * train_ratio)
    test_len = max(total_len - train_len, 0)
    target_dim = 1
    if feature == "M":
        target_dim = 1
    return {
        "total_length": total_len,
        "train_length": train_len,
        "test_length": test_len,
        "target_dim": target_dim,
        "feat_dynamic_real_dim": int(future_feat_dim),
        "past_feat_dynamic_real_dim": int(past_feat_dim),
        "prediction_length": prediction_length,
    }


def clip_context_by_available_history(total_len: int, prediction_length: int, context_length: int) -> int:
    available_history = max(total_len - prediction_length, 0)
    if context_length > available_history:
        logger.warning(
            "上下文长度 {cl} 超过可用历史长度 {ah}；已裁剪。",
            cl=context_length,
            ah=available_history,
        )
        return available_history
    return context_length


def clip_context_for_extrapolation(total_len: int, context_length: int) -> int:
    """用于纯外推预测的上下文裁剪：不保留预测窗口，直接按总长度裁剪。

    - 若 `context_length` 超过序列总长 `total_len`，则裁剪为 `total_len`。
    - 该逻辑仅用于预测接口的“未来外推”，不用于评估。
    """
    if context_length > total_len:
        logger.warning(
            "上下文长度 {cl} 超过序列总长 {tl}；已裁剪。",
            cl=context_length,
            tl=total_len,
        )
        return total_len
    return context_length


def enforce_moirai2_small_pred_len(prediction_length: int) -> int:
    # 对于 moirai-2.0-R-small：patch_size=16，num_predict_token=4，最多预测 64 步
    max_len = 64
    if prediction_length > max_len:
        logger.warning(
            "预测步数 {} 超过 moirai-2.0-R-small 最大值 {}；已裁剪。",
            prediction_length,
            max_len,
        )
        return max_len
    return prediction_length


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不匹配：真实 {y_true.shape}，预测 {y_pred.shape}")
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    return mse, mae
