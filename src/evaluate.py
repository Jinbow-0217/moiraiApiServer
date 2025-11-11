#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2025/11/11 15:12:53
@Author  :   kaixinpangpangyu
@Version :   1.0
@Contact :   wangjinbo_0217@163.com
@Motto   :   Innovate Today
'''


from typing import Dict

import numpy as np
from loguru import logger
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from .utils import (
    resolve_moirai2_local_path,
    load_csv_target,
    compute_metadata,
    clip_context_by_available_history,
    enforce_moirai2_small_pred_len,
    compute_metrics,
    make_sliding_context_and_labels,
    build_listdataset_from_contexts,
)


def _init_moirai2(metadata: Dict, context_length: int) -> tuple[Moirai2Forecast, int]:
    local_dir = resolve_moirai2_local_path()
    used_ctx = clip_context_by_available_history(
        total_len=metadata["total_length"],
        prediction_length=metadata["prediction_length"],
        context_length=context_length,
    )

    logger.info(
        "初始化 Moirai2：预测步数={}，上下文长度={}（目标维度={}）",
        metadata["prediction_length"],
        used_ctx,
        metadata["target_dim"],
    )

    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(local_dir),
        prediction_length=metadata["prediction_length"],
        context_length=used_ctx,
        target_dim=metadata["target_dim"],
        feat_dynamic_real_dim=metadata["feat_dynamic_real_dim"],
        past_feat_dynamic_real_dim=metadata["past_feat_dynamic_real_dim"],
    )
    return model, used_ctx


def evaluate_dataset_mse_mae(
    csv_path: str,
    target_column: str,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    freq: str,
    train_ratio: float,
) -> Dict:
    # 准备数据（滑动窗口：覆盖全序列）
    raw = load_csv_target(csv_path, target_column)
    prediction_length = enforce_moirai2_small_pred_len(prediction_length)
    metadata = compute_metadata(raw, train_ratio, prediction_length)

    # 初始化模型与上下文长度（用于窗口宽度）
    model, used_ctx = _init_moirai2(metadata, context_length)

    # 生成滑动窗口的上下文与标签对；步长默认为 prediction_length
    contexts, labels = make_sliding_context_and_labels(
        target=raw,
        context_length=used_ctx,
        prediction_length=prediction_length,
        step=prediction_length,
    )
    windows = len(contexts)
    if windows == 0:
        raise RuntimeError("无有效滑动窗口；序列长度不足以评估。")

    logger.info(
        "使用滑动窗口评估：窗口数={}，上下文长度={}，预测步数={}",
        windows,
        used_ctx,
        prediction_length,
    )

    # 将所有上下文打包为数据集，批量预测
    # 为滑动窗口数据集尝试使用 CSV 日期列作为每个窗口的起始时间戳；步长为 prediction_length
    context_ds = build_listdataset_from_contexts(
        contexts,
        freq=freq,
        csv_path=csv_path,
        date_column="date",
        step=prediction_length,
    )
    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = list(predictor.predict(context_ds))
    if len(forecasts) != windows:
        raise RuntimeError(
            f"预测结果数量不匹配：{len(forecasts)} 与窗口数 {windows}。"
        )

    # 提取每个窗口的预测中位数（若无分位数，则退化为均值或样本均值）
    preds = []
    for fc in forecasts:
        if hasattr(fc, "quantile"):
            p = np.array(fc.quantile(0.5))
        elif hasattr(fc, "mean") and fc.mean is not None:
            p = np.array(fc.mean)
        elif hasattr(fc, "samples") and fc.samples is not None:
            p = np.array(fc.samples).mean(axis=0)
        else:
            raise RuntimeError("不支持的预测对象格式。")
        preds.append(p.squeeze())

    # 展平聚合所有预测与标签，得到全序列评估
    y_pred = np.concatenate([p.reshape(-1) for p in preds], axis=0)
    y_true = np.concatenate([l.reshape(-1) for l in labels], axis=0)

    mse, mae = compute_metrics(y_true, y_pred)

    return {
        "mse": mse,
        "mae": mae,
        "windows": int(windows),
        "points": int(y_true.shape[0]),
        "usedPredictionLength": int(prediction_length),
        "usedContextLength": int(used_ctx),
        "targetDim": int(metadata["target_dim"]),
        "meta": metadata,
    }