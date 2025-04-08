import pandas as pd
import openai
import os
import asyncio
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np


# 配置设置
BATCH_SIZE = 20  # 根据 API 限制调整
EMBEDDING_MODEL = "text-embedding-3-small"


# Load knowledge database
rag_db = pd.read_csv("./dataset/kd.csv")
rag_data = rag_db[['ID', 'Content']].dropna()

# Load knowledge database
rag_data['Content'] = rag_data['Content'].str.lower()
documents = rag_data['Content'].tolist()
doc_ids = rag_data['ID'].tolist()

llm_choice = "gpt"
openai.api_key = os.environ["OPENAI_API_KEY"]




embeddings_path = './dataset/doc_embeddings.npy'

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
async def get_batch_embeddings(batch_texts):
    # 初始化异步客户端
    client = AsyncOpenAI()
    """异步批量获取embedding并自动重试"""
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts,
            dimensions=1536  # 可选512/1024/1536
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"批量处理失败，错误: {str(e)}")
        raise

async def process_all_batches(documents):
    """异步处理所有批次"""
    tasks = []
    
    # 分割为批次
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        task = get_batch_embeddings(batch)
        tasks.append(task)
        print(f"已提交批次 {i//BATCH_SIZE + 1}/{(len(documents)-1)//BATCH_SIZE + 1}")

    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 合并结果
    embeddings = []
    for batch_result in results:
        if isinstance(batch_result, Exception):
            print(f"遇到错误批次: {str(batch_result)}")
            embeddings.extend([None]*BATCH_SIZE)  # 标记失败批次
        else:
            embeddings.extend(batch_result)
    
    return embeddings[:len(documents)]  # 对齐数据长度

def main():
    
    # 异步处理
    doc_embeddings = asyncio.run(process_all_batches(documents))

    doc_embeddings = np.vstack(doc_embeddings)
    
    # 保存结果（建议使用parquet格式节省空间）
    np.save(embeddings_path, doc_embeddings)  # Save embeddings to disk

    print(f"处理完成！有效embedding数量：{documents.notnull().sum()}/{len(documents)}")

if __name__ == "__main__":
    main()