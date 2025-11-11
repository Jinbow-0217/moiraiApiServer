## Moirai API Server（基于 moirai-2.0-small）

[Moirai接口服务示意图](figures/moirai_api_server.png)

## 概述
- 基于 Salesforce Moirai 2.0 小型模型开发的时序预测 API 服务。
- 当前仅支持单变量预测（S 类型）；后续将拓展到 MS 和 M 类型，并丰富更多 Moirai 系列模型。
- 默认使用本地模型快照目录 `bin/moirai-2.0-R-small/`（需包含 `config.json` 与 `model.safetensors`）。

## 快速开始
- 环境要求：Python 3.10+。
- 安装依赖：`pip install -r requirements.txt`
- 启动服务：`uvicorn app:app --host 0.0.0.0 --port 8217`
- 在线文档：访问 `http://localhost:8217/docs`（Swagger）或 `http://localhost:8217/redoc`。

## 使用流程
- 先通过接口 `/evaluate` 进行评估确认数据与参数是否合适。
- 评估合适后，再调用接口 `/forecast` 进行外推预测。
- 若运行报错，使用接口 `/download-log` 查询对应日志（日志文件名为 `taskCode.log`）。

## 接口示例
- `POST /evaluate`
  - 请求体（JSON，驼峰命名）：
    - `taskCode`、`datasetPath`、`targetColumn`、`contextLength`、`predictionLength`、`batchSize`、`freq`、`trainRatio`
  - 示例：
    - `curl -X POST http://localhost:8217/evaluate -H "Content-Type: application/json" -d '{"taskCode":"etth1-eval","datasetPath":"datasets/ETT-small/ETTh1.csv","targetColumn":"OT","contextLength":1680,"predictionLength":64,"batchSize":8,"freq":"H","trainRatio":0.8}'`
  - 响应：`mse`、`mae`、`usedPredictionLength`、`usedContextLength`、`targetDim`、`meta`。

- `POST /forecast`
  - 在 `/evaluate` 的字段基础上，增加 `lowerQuantile`、`upperQuantile`
  - 示例：
    - `curl -X POST http://localhost:8217/forecast -H "Content-Type: application/json" -d '{"taskCode":"etth1-forecast","datasetPath":"datasets/ETT-small/ETTh1.csv","targetColumn":"OT","contextLength":1680,"predictionLength":64,"batchSize":8,"lowerQuantile":0.1,"upperQuantile":0.9,"freq":"H","trainRatio":0.8}'`
  - 响应：`median`、`lower`、`upper`、`usedPredictionLength`、`usedContextLength`、`targetDim`、`meta`。

- `GET /download-log`
  - 查询参数：`taskCode`、`password`（默认 `moirai`）。
  - 示例：`curl "http://localhost:8217/download-log?taskCode=etth1-forecast&password=moirai"`

- `GET /health`
  - 健康检查：返回 `{"status": "ok"}`。

## 配置
- 可选环境变量（前缀 `MOIRAI_`）：
  - `MOIRAI_MODELS_DIRNAME`：模型目录名，默认 `bin`
  - `MOIRAI_MOIRAI2_LOCAL_DIRNAME`：Moirai2 本地快照目录名，默认 `moirai-2.0-R-small`
  - `MOIRAI_LOG_DOWNLOAD_PASSWORD`：日志下载密码，默认 `moirai`

## 日志与数据
- 日志存放于项目根目录 `logs/` 下，文件名为 `taskCode.log`。
- 示例数据可放在 `datasets/ETT-small/`（如 `ETTh1.csv`）。

## Docker 部署
- 构建镜像：`docker build -t moirai-api:latest .`
- 运行容器（映射端口与挂载数据/日志）：
  - `docker run --rm -p 8217:8217 \
    -e MOIRAI_LOG_DOWNLOAD_PASSWORD=moirai \
    -v $(pwd)/datasets:/app/datasets \
    -v $(pwd)/logs:/app/logs \
    moirai-api:latest`
- 说明：镜像已包含 `bin/moirai-2.0-R-small/` 模型快照；数据与日志建议通过卷挂载管理。

## 接口文档
- 详见 `docs/Interface.md`（与 `app.py` 保持一致的字段与示例）。

## 联系方式
- Email：`wangjinbo_0217@163.com`