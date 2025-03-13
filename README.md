# CS450-Winter

## **Repository Update (3/12/2025) - Jinwei**
### **📌 Key Changes**
- **Repository Cleanup:**  
  - All work from this semester has been moved to the `/2025_Winter` directory.
  - Each test (`test_1`, `test_2`, `test_3`, etc.) has its corresponding folder inside `/2025_Winter`, containing `.npy` files for the respective test runs.
  - All required **input files** are now stored in `/dataset`, except for the **testing data**, which is located in `/2025_Winter` as `784_structural_input.csv`.
  - Each `/input_n/` directory contains the **generated results** from the current test run, stored as `/result_784_structural`.

---

### **🧪 Tests Overview**
#### **Dataset: 784 Structured**
| Test     | RAG Version | Re-ranking Method                                      | Prompt Version |
|----------|------------|--------------------------------------------------------|---------------|
| **Baseline** | No RAG     | No re-ranking                                        | No prompt    |
| **test#1**   | RAG 1.0    | No tag-based re-ranking                              | Prompt 3.0   |
| **test#2**   | RAG 2.0    | **Method 1: Re-ranking by frequency of the tags**    | Prompt 3.0   |
| **test#3**   | RAG 2.0    | **Method 2: Re-ranking by tag similarity**          | Prompt 3.0   |
| **test#4**   | RAG 2.0    | **Method 1: Re-ranking by frequency of the tags**    | Prompt 4.0   |
| **test#5**   | RAG 2.0    | **Method 2: Re-ranking by tag similarity**          | Prompt 4.0   |

---

### **📜 Prompt Versions**
| Prompt Version | Description |
|---------------|------------|
| **Prompt 1.0** | Jack's old prompt (Fall 2024) – **broken** |
| **Prompt 2.0** | Non-XML format, includes **step-by-step reasoning** |
| **Prompt 3.0** | Uses **XML format**, only generates **YAML output**, no step-by-step |
| **Prompt 4.0** | Based on 3.0, goal is to improve **faithfulness**, making sure LLM answer strictly follows RAG retrieved results |

---

### **🔍 RAG Versions**
| RAG Version | Description |
|------------|-------------|
| **RAG 1.0** | No tag-based retrieval |
| **RAG 2.0** | Supports **tag-based retrieval** |

---
