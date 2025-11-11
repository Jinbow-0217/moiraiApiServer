#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2025/11/11 15:11:53
@Author  :   kaixinpangpangyu
@Version :   1.0
@Contact :   wangjinbo_0217@163.com
@Motto   :   Innovate Today
'''

# Moirai API Server 全局配置（可扩展）

import os


class AppSettings:
    """应用配置，支持通过环境变量覆盖默认值。

    环境变量（可选）：
    - `MOIRAI_MODELS_DIRNAME`：模型目录名，默认 `bin`
    - `MOIRAI_MOIRAI2_LOCAL_DIRNAME`：Moirai2 本地快照目录名，默认 `moirai-2.0-R-small`
    - `MOIRAI_LOG_DOWNLOAD_PASSWORD`：日志下载接口密码，默认 `moirai`
    """

    def __init__(self) -> None:
        self.models_dirname: str = os.getenv("MOIRAI_MODELS_DIRNAME", "bin")
        self.moirai2_local_dirname: str = os.getenv(
            "MOIRAI_MOIRAI2_LOCAL_DIRNAME", "moirai-2.0-R-small"
        )
        self.log_download_password: str = os.getenv("MOIRAI_LOG_DOWNLOAD_PASSWORD", "moirai")


# 实例化配置（用于运行时读取）
settings = AppSettings()