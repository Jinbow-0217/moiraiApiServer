#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   forecast.py
@Time    :   2025/11/11 16:10:16
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
    load_csv_target_and_covariates,
    build_context_only_listdataset,
    compute_metadata,
    clip_context_for_extrapolation,
    enforce_moirai2_small_pred_len,
)


def _init_moirai2(metadata: Dict, context_length: int) -> tuple[Moirai2Forecast, int]:
    local_dir = resolve_moirai2_local_path()
    # 纯外推模式：只根据总长度裁剪上下文，不保留预测窗口
    used_ctx = clip_context_for_extrapolation(
        total_len=metadata["total_length"],
        context_length=context_length,
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


def forecast_with_quantiles(
    csv_path: str,
    target_column: str,
    feature: str,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    lower_q: float,
    upper_q: float,
    freq: str,
    train_ratio: float,
):
    # 准备数据
    if feature == "MS":
        raw, covs, cov_cols = load_csv_target_and_covariates(csv_path, target_column, "date")
    else:
        raw = load_csv_target(csv_path, target_column)
        covs = None
    prediction_length = enforce_moirai2_small_pred_len(prediction_length)
    metadata = compute_metadata(
        raw,
        train_ratio,
        prediction_length,
        feature=feature,
        past_feat_dim=(covs.shape[0] if covs is not None else 0),
        future_feat_dim=0,
    )

    # 初始化并执行预测（仅使用末尾上下文进行未来外推）
    model, used_ctx = _init_moirai2(metadata, context_length)
    predictor = model.create_predictor(batch_size=batch_size)
    # 单窗口外推：尝试使用 CSV 日期列作为起始时间戳；失败则降级为占位时间戳
    context_ds = build_context_only_listdataset(
        raw,
        freq=freq,
        used_ctx=used_ctx,
        csv_path=csv_path,
        date_column="date",
        past_covs=covs if covs is not None and covs.size > 0 else None,
    )
    forecasts = list(predictor.predict(context_ds))
    if not forecasts:
        raise RuntimeError("未产生任何预测结果。")
    fc = forecasts[0]

    # 提取分位数与中位数
    if hasattr(fc, "quantile"):
        median = np.array(fc.quantile(0.5)).tolist()
        lower = np.array(fc.quantile(lower_q)).tolist()
        upper = np.array(fc.quantile(upper_q)).tolist()
    else:
        # 兜底：以均值作为中位数；若存在样本，则用 ±1 标准差作为上下界
        if hasattr(fc, "mean") and fc.mean is not None:
            m = np.array(fc.mean)
        elif hasattr(fc, "samples") and fc.samples is not None:
            m = np.array(fc.samples).mean(axis=0)
        else:
            raise RuntimeError("不支持的预测对象：不存在分位数、均值或样本。")
        median = m.squeeze().tolist()
        if hasattr(fc, "samples") and fc.samples is not None:
            s = np.array(fc.samples)
            std = s.std(axis=0)
            lower = (m - std).squeeze().tolist()
            upper = (m + std).squeeze().tolist()
        else:
            lower = median
            upper = median

    return {
        "median": median,
        "lower": lower,
        "upper": upper,
        "usedPredictionLength": int(prediction_length),
        "usedContextLength": int(used_ctx),
        "targetDim": int(metadata["target_dim"]),
        "meta": metadata,
    }
