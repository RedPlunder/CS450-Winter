import json
import os
import asyncio
from openai import AsyncOpenAI
import openai

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# Set your OPENAI API KEY here.
openai.api_key = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI()

json_file_name = "test_13processed_data.json"
CONCURRENCY_LIMIT = 10  # 根据你的网络情况调整并发数

async def refer_standardization(reference_answer):

    prompt = f"""
    You are an expert in the K8s field. Now, I will provide you with standard answers to some questions. 
    Please rewrite the answers based on the original content. The answer that requires rewriting is divided into two parts.
    The first part is the YAML file code, and the second part is the explanation.
    For example:
    <standard answer>
    "i think that this pr contains the change you're asking about.\n`ingress` and `ingressclass` resources have graduated to `networking.k8s.io/v1`. ingress and ingressclass types in the `extensions/v1beta1` and `networking.k8s.io/v1beta1` api versions are deprecated and will no longer be served in 1.22+. persisted objects can be accessed via the `networking.k8s.io/v1` api. notable changes in v1 ingress objects (v1beta1 field names are unchanged):\n* `spec.backend` -&gt; `spec.defaultbackend`\n* `servicename` -&gt; `service.name`\n* `serviceport` -&gt; `service.port.name` (for string values)\n* `serviceport` -&gt; `service.port.number` (for numeric values)\n* `pathtype` no longer has a default value in v1; &quot;exact&quot;, &quot;prefix&quot;, or &quot;implementationspecific&quot; must be specified\nother ingress api updates:\n* backends can now be resource or service backends\n* `path` is no longer required to be a valid regular expression\n\nif you look in the 1.19 ingress doc, it looks like the new syntax would be:\napiversion: networking.k8s.io/v1\nkind: ingress\nmetadata:\n  name: minimal-ingress\n  annotations:\n    nginx.ingress.kubernetes.io/rewrite-target: /\nspec:\n  rules:\n  - http:\n      paths:\n      - path: /testpath\n        pathtype: prefix\n        backend:\n          service:\n            name: test\n            port:\n              number: 80\n\ni unfortunately don't have a 1.19 cluster to test myself, but i think this is what you're running into."
    </standard answer>
    <rewrite answer>
    ```yaml
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
    name: minimal-ingress
    annotations:
        nginx.ingress.kubernetes.io/rewrite-target: /
    spec:
    rules:
    - http:
        paths:
        - path: /testpath
            pathType: Prefix         # 必须显式声明（注意PascalCase）
            backend:
            service:
                name: test
                port: 
                number: 80        # 数字端口号使用number字段
    ```
    ### Explanation
    i think that this pr contains the change you're asking about. `ingress` and `ingressclass` resources have graduated to `networking.k8s.io/v1`. ingress and ingressclass types in the `extensions/v1beta1` and `networking.k8s.io/v1beta1` api versions are deprecated and will no longer be served in 1.22+. persisted objects can be accessed via the `networking.k8s.io/v1` api. notable changes in v1 ingress objects (v1beta1 field names are unchanged):
    * `spec.backend` -&gt; `spec.defaultbackend`
    * `servicename` -&gt; `service.name`
    * `serviceport` -&gt; `service.port.name` (for string values)
    * `serviceport` -&gt; `service.port.number` (for numeric values)
    </rewrite answer>

    standard answers is below:
    {reference_answer}
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"请求出错: {str(e)}")
        return reference_answer  # 返回原答案避免数据丢失

async def process_items(data):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async def process_with_semaphore(item):
        async with semaphore:
            print(f"Start rewrite answer {data.index(item)}...")
            item["reference_answer"] = await refer_standardization(item["reference_answer"])
            return item
    
    tasks = [process_with_semaphore(item) for item in data]
    return await asyncio.gather(*tasks)

async def main():
    # 读取数据
    with open(json_file_name, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    # 异步处理所有项目
    processed_data = await process_items(data)
    
    # 保存结果
    with open("test_131processed_data.json", "w", encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())