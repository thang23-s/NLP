import redis
import csv
import numpy as np
import pandas as pd
import datetime

# Kết nối Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Đọc file dữ liệu
df = pd.read_csv("merged_data.csv")

# Tạo chỉ mục Redis cho truy vấn semantic
def create_index():
    try:
        r.execute_command(
            "FT.CREATE", "question_idx",
            "ON", "HASH",
            "PREFIX", "1", "q:",
            "SCHEMA",
            "question", "TEXT",
            "short_answer", "TEXT",
            "long_answer", "TEXT",
            "time_exec", "NUMERIC",
            "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32", "DIM", "384", "DISTANCE_METRIC", "COSINE"
        )
    except redis.exceptions.ResponseError as e:
        if "Index already exists" not in str(e):
            raise

# Lưu dữ liệu từng dòng vào Redis
def save_to_redis(idx, row):
    question = row.get("question", "")
    short_answer = row.get("short_answer", "")
    long_answer = row.get("long_answer", "")
    embedding = row[[f"emb_{i}" for i in range(384)]].astype(np.float32).values
    embedding_bytes = embedding.tobytes()

    redis_key = f"q:{idx}"
    r.hset(redis_key, mapping={
        b"question": question.encode("utf-8"),
        b"short_answer": short_answer.encode("utf-8"),
        b"long_answer": long_answer.encode("utf-8"),
        b"time_exec": int(datetime.date.today().strftime("%Y%m%d")),
        b"embedding": embedding_bytes
    })

# Chạy tạo index và ghi dữ liệu
create_index()
for idx, row in df.iterrows():
    save_to_redis(idx, row)
