import os
import pandas as pd
import numpy as np
import faiss
import asyncio
import openai
from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoModel
from tenacity import retry, wait_random_exponential, stop_after_attempt

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"

# 全局配置
llm_choice = "gpt"
openai.api_key = "your-api-key"  # 建议放到环境变量而不是硬编码
BATCH_SIZE = 20
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_PATH = './dataset/doc_embeddings.npy'
INPUT_CSV = 'new_input_data.csv'
# =========== GLOBAL CLIENTS ===========
sync_client = openai.OpenAI(api_key=openai.api_key)  # 同步client，用于同步embedding
async_client = AsyncOpenAI(api_key=openai.api_key)   # 异步client，用于异步生成response

# 加载知识库
rag_db = pd.read_csv("./kd/kd.csv")[['ID', 'Content']].dropna()
rag_db['Content'] = rag_db['Content'].str.lower()
documents = rag_db['Content'].tolist()
doc_ids = rag_db['ID'].tolist()

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
async def get_batch_embeddings(batch_texts):
    client = openai.OpenAI(api_key="your-api-key")  # 显式传入
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=batch_texts,
    )
    return [item.embedding for item in response.data]

async def process_all_batches(documents):
    tasks = [
        get_batch_embeddings(documents[i:i+BATCH_SIZE])
        for i in range(0, len(documents), BATCH_SIZE)
    ]
    print(f"提交 {len(tasks)} 个批次异步任务。")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    embeddings = []
    for result in results:
        if isinstance(result, Exception):
            print(f"批次处理失败: {str(result)}")
            embeddings.extend([None] * BATCH_SIZE)
        else:
            embeddings.extend(result)
    return np.vstack([emb for emb in embeddings if emb is not None])

def load_or_create_embeddings():
    if os.path.exists(EMBEDDING_PATH):
        print("从文件加载已有embedding...")
        return np.load(EMBEDDING_PATH, mmap_mode='r')
    else:
        embeddings = asyncio.run(process_all_batches(documents))
        np.save(EMBEDDING_PATH, embeddings)
        return embeddings

doc_embeddings = load_or_create_embeddings()

# 建立 FAISS 索引
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)
print("FAISS indexing complete.")

def truncate_text(text, max_chars=4096):
    return text if len(text) <= max_chars else text[:max_chars]

def get_embedding(text):
    try:
        response = sync_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            # dimensions=EMBEDDING_DIM
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding错误: {str(e)}")
        return None

def embed(texts):
    """同步小批量embed，防止内存爆"""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        batch_emb = [get_embedding(text) for text in batch]
        all_embeddings.append(batch_emb)
    return np.vstack(all_embeddings)

def retrieve_documents(queries, k=3):
    query_embeddings = embed([truncate_text(q.strip().lower(), max_chars=23552) for q in queries])
    distances, indices = index.search(np.array(query_embeddings), k)
    
    results = []
    for query_indices in indices:
        result = {
            "ids": [doc_ids[i] for i in query_indices],
            "contents": [documents[i] for i in query_indices]
        }
        results.append(result)
    return results
async def generate_responses(
    title_queries,
    body_queries,
    contexts,
    previous_answers=None
):
    async def call_openai_multi_turn(messages):
        for attempt in range(3):
            try:
                response = await async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=4090,
                    temperature=0
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                print(f"速率限制，重试 {attempt+1}/3")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"调用OpenAI异常: {e}")
                return f"API Error: {e}"
        return "API Error: Max retries exceeded."

    prompts = []
    for title, body, context, prev_ans in zip(
        title_queries, body_queries, contexts, previous_answers or [None]*len(title_queries)
    ):
        # Round 1: first-time answering
        if prev_ans is None:
            messages = [
                {"role": "system", "content": "You are a helpful technical assistant."},
                {
                    "role": "user",
                    "content": (
                        f"<retrieved_knowledge><![CDATA[{context}]]></retrieved_knowledge>\n"
                        f"<user_query><title>{title}</title><body>{body}</body></user_query>\n"
                        f"Please answer the user's question accurately and concisely."
                    )
                }
            ]
        else:
            # Round 2: refine based on previous response
            messages = [
                {"role": "system", "content": "You are a reliable technical assistant."},
                {
                    "role": "user",
                    "content": (
                        f"<retrieved_knowledge><![CDATA[{context}]]></retrieved_knowledge>\n"
                        f"<user_query><title>{title}</title><body>{body}</body></user_query>\n"
                        f"Please answer the user's question."
                    )
                },
                {"role": "assistant", "content": prev_ans},
                {
                    "role": "user",
                    "content": (
                        ## v0
                        # "Please review your previous answer using the retrieved knowledge. "
                        # "If it contains mistakes or is incomplete, fix them. If it is correct, rephrase it more clearly. "
                        # "Only modify what's necessary; do not change formatting unnecessarily."
                        ## v1
                        "Please review your previous answer using the retrieved knowledge. Ensure that all key configuration points in your answer are directly grounded in the retrieved context. Identify which parts of your answer correspond to the retrieved context, and revise any incorrect, incomplete, or unsupported parts. If any part of your answer reflects general practices learned during pretraining that contradict the retrieved context, prioritize the context and revise accordingly. Do not introduce solutions or explanations that are not supported by the context. Keep the formatting and structure of your original answer unchanged—only modify the content where necessary for accuracy and clarity."
                    )
                }
            ]

        prompts.append(messages)

    # Run all calls in parallel
    return await asyncio.gather(*[call_openai_multi_turn(msg) for msg in prompts])


async def process_questions(file_path, batch_size=40):
    df = pd.read_csv(file_path, encoding="utf-8")
    context_cols = [
        f"{llm_choice}_Top_1_Context",
        f"{llm_choice}_Top_2_Context",
        f"{llm_choice}_Top_3_Context",
        f"{llm_choice}_Merged_Contexts",
        f"{llm_choice}_Generated_Response",
        f"{llm_choice}_Refined_Response",
        f"{llm_choice}_Context_IDs",
    ]
    for col in context_cols:
        if col not in df.columns:
            df[col] = ""

    # 仅处理未完成的样本（未refined的）
    unanswered_df = df[df[f"{llm_choice}_Refined_Response"] == ""]

    for start in range(0, len(unanswered_df), batch_size):
        batch = unanswered_df.iloc[start:start+batch_size]

        title_queries = batch["Question Title"].tolist()
        body_queries = batch["Question Body"].tolist()

        results = retrieve_documents(body_queries, k=3)

        top_1 = [truncate_text(r["contents"][0]) if len(r["contents"]) > 0 else "" for r in results]
        top_2 = [truncate_text(r["contents"][1]) if len(r["contents"]) > 1 else "" for r in results]
        top_3 = [truncate_text(r["contents"][2]) if len(r["contents"]) > 2 else "" for r in results]
        merged_contexts = [" ".join(truncate_text(c) for c in r["contents"]) for r in results]
        context_ids = [", ".join(map(str, r["ids"])) for r in results]

        # 第一轮回答
        first_responses = await generate_responses(
            title_queries,
            body_queries,
            merged_contexts,
            previous_answers=None  # 第一轮调用
        )

        # 第二轮 refine
        refined_responses = await generate_responses(
            title_queries,
            body_queries,
            merged_contexts,
            previous_answers=first_responses
        )

        df.loc[batch.index, f"{llm_choice}_Generated_Response"] = first_responses
        df.loc[batch.index, f"{llm_choice}_Refined_Response"] = refined_responses
        df.loc[batch.index, f"{llm_choice}_Merged_Contexts"] = merged_contexts
        df.loc[batch.index, f"{llm_choice}_Top_1_Context"] = top_1
        df.loc[batch.index, f"{llm_choice}_Top_2_Context"] = top_2
        df.loc[batch.index, f"{llm_choice}_Top_3_Context"] = top_3
        df.loc[batch.index, f"{llm_choice}_Context_IDs"] = context_ids

        print(f"已处理 {start + len(batch)} / {len(unanswered_df)} 个问题")

    output_file = "test_verification_results_v1.csv"

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"所有问题已处理并保存到 {output_file}")



if __name__ == "__main__":
    asyncio.run(process_questions(INPUT_CSV))
