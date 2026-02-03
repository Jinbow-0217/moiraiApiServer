#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   app.py
@Time    :   2025/11/11 15:12:27
@Author  :   kaixinpangpangyu
@Version :   1.0
@Contact :   wangjinbo_0217@163.com
@Motto   :   Innovate Today
'''


import os
from typing import Optional, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from loguru import logger

from src.evaluate import evaluate_dataset_mse_mae
from src.forecast import forecast_with_quantiles
from settings.config import settings


app = FastAPI(title="Moirai API Server", version="0.1.0")


class EvaluateRequest(BaseModel):
    taskCode: str = Field(..., description="任务代码（用于日志文件命名）")
    datasetPath: str = Field(..., description="CSV 文件路径")
    targetColumn: str = Field(..., description="目标列名")
    feature: Literal["S", "MS", "M"] = Field("S", description="特征类型：S（单变量）、MS（多变量协变量-单目标）、M（多目标）")
    contextLength: int = Field(1680, description="上下文长度")
    predictionLength: int = Field(64, description="预测步数（moirai-2.0-R-small最大建议64）")
    batchSize: int = Field(8, description="预测批大小")
    freq: str = Field("H", description="时间频率，例如 H, 15min, D")
    trainRatio: float = Field(0.8, description="训练比例（仅用于元数据标注）")



class EvaluateResponse(BaseModel):
    mse: float
    mae: float
    windows: int | None = Field(default=None, description="滑动窗口数量（可选）")
    points: int | None = Field(default=None, description="参与评估的总点数（可选）")
    usedPredictionLength: int
    usedContextLength: int
    targetDim: int
    meta: dict


class ForecastRequest(BaseModel):
    taskCode: str = Field(..., description="任务代码（用于日志文件命名）")
    datasetPath: str = Field(..., description="CSV 文件路径")
    targetColumn: str = Field(..., description="目标列名")
    feature: Literal["S", "MS", "M"] = Field("S", description="特征类型：S（单变量）、MS（多变量协变量-单目标）、M（多目标）")
    contextLength: int = Field(1680, description="上下文长度")
    predictionLength: int = Field(64, description="预测步数（moirai-2.0-R-small最大建议64）")
    batchSize: int = Field(8, description="预测批大小")
    lowerQuantile: float = Field(0.1, description="下分位（例如 0.1）")
    upperQuantile: float = Field(0.9, description="上分位（例如 0.9）")
    freq: str = Field("H", description="时间频率，例如 H, 15min, D")
    trainRatio: float = Field(0.8, description="训练比例（仅用于元数据标注）")


class ForecastResponse(BaseModel):
    median: List[float]
    lower: List[float]
    upper: List[float]
    usedPredictionLength: int
    usedContextLength: int
    targetDim: int
    meta: dict


@app.get("/health")
def health():
    return {"status": "ok"}
    

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    # 为本次请求创建独立日志文件
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{req.taskCode}.log")
    sink_id = logger.add(log_file, enqueue=True, backtrace=False, diagnose=False, level="INFO")
    try:
        logger.info("评估接口开始，taskCode={}，请求载荷={}", req.taskCode, req.model_dump())
        result = evaluate_dataset_mse_mae(
            csv_path=req.datasetPath,
            target_column=req.targetColumn,
            feature=req.feature,
            context_length=req.contextLength,
            prediction_length=req.predictionLength,
            batch_size=req.batchSize,
            freq=req.freq,
            train_ratio=req.trainRatio,
        )
        logger.info("评估成功，taskCode={}，指标摘要：mse={}，mae={}", req.taskCode, result.get("mse"), result.get("mae"))
        return EvaluateResponse(**result)
    except Exception as e:
        logger.exception("评估失败，taskCode={}：{}", req.taskCode, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            logger.remove(sink_id)
        except Exception:
            pass


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 为本次请求创建独立日志文件
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{req.taskCode}.log")
    sink_id = logger.add(log_file, enqueue=True, backtrace=False, diagnose=False, level="INFO")
    try:
        logger.info("预测接口开始，taskCode={}，请求载荷={}", req.taskCode, req.model_dump())
        result = forecast_with_quantiles(
            csv_path=req.datasetPath,
            target_column=req.targetColumn,
            feature=req.feature,
            context_length=req.contextLength,
            prediction_length=req.predictionLength,
            batch_size=req.batchSize,
            lower_q=req.lowerQuantile,
            upper_q=req.upperQuantile,
            freq=req.freq,
            train_ratio=req.trainRatio,
        )
        logger.info("预测成功，taskCode={}，使用的预测步数={}，上下文长度={}", req.taskCode, result.get("usedPredictionLength"), result.get("usedContextLength"))
        return ForecastResponse(**result)
    except Exception as e:
        logger.exception("预测失败，taskCode={}：{}", req.taskCode, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            logger.remove(sink_id)
        except Exception:
            pass


@app.get("/download-log")
def download_log(taskCode: str, password: str):
    """按 taskCode 下载日志文件，需提供正确密码。"""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_file = os.path.join(logs_dir, f"{taskCode}.log")
    try:
        logger.info("日志下载开始，taskCode={}", taskCode)
        if password != settings.log_download_password:
            logger.warning("日志下载失败，密码错误，taskCode={}", taskCode)
            raise HTTPException(status_code=403, detail="密码错误")
        if not os.path.isfile(log_file):
            logger.warning("日志文件不存在，taskCode={}，路径={}", taskCode, log_file)
            raise HTTPException(status_code=404, detail="日志文件不存在")
        logger.info("日志下载成功，taskCode={}，路径={}", taskCode, log_file)
        return FileResponse(log_file, filename=f"{taskCode}.log", media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("日志下载异常，taskCode={}：{}", taskCode, e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 作为脚本运行时，使用 uvicorn 启动服务在端口 8217
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8217, reload=False)
