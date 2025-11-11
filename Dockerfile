FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（如需编译或科学计算库，可按需扩展）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt ./
COPY app.py ./
COPY settings ./settings
COPY src ./src
COPY docs ./docs
COPY bin ./bin

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8217

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8217"]