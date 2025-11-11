# Moirai API 接口文档

- 基于 `moirai-2.0-small` 的时序预测服务，当前支持单变量预测（S 类型），后续将拓展 MS/M 类型与更多模型。
- 推荐流程：先调用 `/evaluate` 评估数据与参数是否合适；确认后通过 `/forecast` 外推预测；若报错可用 `/download-log` 下载日志排查。

## 基础信息
 - 基地址：`http://localhost:8217`
 - 健康检查：`GET /health` 返回 `{"status":"ok"}`

## 评估接口
 - 路径：`POST /evaluate`
 - 请求体（JSON，驼峰命名）：
   - `taskCode`：任务代码（也用于日志文件名，如 `etth1-eval`）
   - `datasetPath`：CSV 文件路径（如 `datasets/ETT-small/ETTh1.csv`）
   - `targetColumn`：目标列名（如 `OT`）
   - `contextLength`：上下文长度（默认 1680）
   - `predictionLength`：预测步数（小模型建议 ≤64）
   - `batchSize`：预测批大小（默认 8）
   - `freq`：时间频率（如 `H`、`15min`、`D`）
   - `trainRatio`：训练比例（仅用于元数据标注）
 - 示例：
   - `curl -X POST http://localhost:8217/evaluate -H "Content-Type: application/json" -d '{"taskCode":"etth1-eval","datasetPath":"datasets/ETT-small/ETTh1.csv","targetColumn":"OT","contextLength":1680,"predictionLength":64,"batchSize":8,"freq":"H","trainRatio":0.8}'` 
 - 响应字段（示例）：
   - `mse`、`mae`、`windows`、`points`、`usedPredictionLength`、`usedContextLength`、`targetDim`、`meta`

## 预测接口

 - 路径：`POST /forecast`
 - 请求体（JSON，驼峰命名）：
   - 与 `/evaluate` 相同字段，另加：
   - `lowerQuantile`、`upperQuantile`：分位数（如 0.1 / 0.9）
 - 示例：
   - `curl -X POST http://localhost:8217/forecast -H "Content-Type: application/json" -d '{"taskCode":"etth1-forecast","datasetPath":"datasets/ETT-small/ETTh1.csv","targetColumn":"OT","contextLength":1680,"predictionLength":64,"batchSize":8,"lowerQuantile":0.1,"upperQuantile":0.9,"freq":"H","trainRatio":0.8}'`
 - 响应字段（示例）：
   - `median`、`lower`、`upper`、`usedPredictionLength`、`usedContextLength`、`targetDim`、`meta`

## 日志下载
 - 路径：`GET /download-log`
 - 查询参数：
   - `taskCode`：任务代码（日志文件名）
   - `password`：下载密码（默认 `moirai`，可通过环境变量修改）
 - 示例：
   - `curl "http://localhost:8217/download-log?taskCode=etth1-forecast&password=moirai"`
 - 错误码：
   - 403：密码错误
   - 404：日志不存在
   - 500：服务异常

## 约束与说明
 - 当前仅支持 S 类型（单变量）；后续将逐步支持 MS 与 M 类型。
 - `predictionLength` 在小模型上可能被自动裁剪到推荐范围（≤64）。
 - 时间轴默认按 `freq` 递增；如 CSV 提供 `date` 列，则优先用该列推定起点。

## 环境变量配置（前缀 `MOIRAI_`）
 - `MOIRAI_MODELS_DIRNAME`：模型目录名，默认 `bin`
 - `MOIRAI_MOIRAI2_LOCAL_DIRNAME`：Moirai2 本地快照目录名，默认 `moirai-2.0-R-small`
 - `MOIRAI_LOG_DOWNLOAD_PASSWORD`：日志下载密码，默认 `moirai`

## 联系方式
 - `wangjinbo_0217@163.com`