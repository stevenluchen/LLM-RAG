# **KDD Cup 2024 RAG Challenge**

This repository contains our implementation for the **KDD Cup 2024 RAG Challenge**, where we iteratively improved a **Retrieval-Augmented Generation (RAG)** system. 

Our final model, built on **LLaMA-3.2-3B**, integrates:
- **Hybrid retrieval** (BM25 + FAISS) for better document grounding.
- **Adaptive query weighting** to improve retrieval effectiveness.
- **Dynamic prompt engineering** to optimize LLM responses.
- **Optimized chunking strategies** to balance recall and precision.

Through rigorous experimentation, we refined `rag_updated.py`, evaluating various retrieval and prompting strategies and comparing them with the baseline `rag_baseline.py`. The improved model demonstrates significant gains in accuracy and exact accuracy based on the official **KDD Cup** evaluation metrics.

The dataset used for this project can be found at [AICrowd](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/).

---

## **Pipeline Overview**
Our pipeline consists of the following modules:

### **Retrieval Strategy**
We experimented with different retrieval techniques:
- **BM25 (Sparse Retrieval)** - Keyword-based document search.
- **FAISS (Dense Retrieval)** - Vector search using semantic similarity.
- **Hybrid Retrieval (BM25 + FAISS)** - Combines both approaches with query-dependent weighting.

### **Chunking Strategy**
- **Paragraph-based chunking** for better context retention.
- **Overlapping sentence chunking** for cases where paragraphs are unavailable.
- **Title extraction** as a separate chunk to improve retrieval.
- **Filtering duplicate and short chunks** to reduce noise.

### **Prompt Engineering**
- **Few-shot learning** - Provides structured examples to guide LLM outputs.
- **Dynamic confidence-based prompting** - Adjusts uncertainty statements based on retrieval confidence.

---

## **Results and Performance Comparison**

The table below compares the **Vanilla Baseline**, the **Initial RAG Baseline**, and our **New Baseline** in terms of key evaluation metrics.

| **Metric**          | **Vanilla Baseline** | **RAG Baseline** | **Updated RAG** |
|---------------------|--------------------|-----------------|----------------|
| **Exact Accuracy**  | 0.0037             | 0.0337          | **0.0412**    |
| **Accuracy**        | 0.1296             | 0.2292          | **0.3101**    |
| **Hallucination**   | 0.1401             | 0.2831          | **0.1900**    |
| **Missing Rate**    | 0.7303             | 0.4876          | **0.4097**    |

### **Raw Counts (Total Questions: 1,335)**
| **Metric**         | **Vanilla Baseline** | **RAG Baseline** | **Updated RAG** |
|-------------------|--------------------|-----------------|----------------|
| **Missing**      | 975                | 651             | **547**       |
| **Correct**      | 173                | 306             | **414**       |
| **Exactly Correct** | 5                 | 45              | **55**        |

### **Key Insights**
- **Significant Accuracy Boost**: The new model improves accuracy from **22.92% → 31.01%**, a **35.3% relative gain** over the previous best.
- **Reduction in Missing Responses**: A notable **15.9% decrease** in missing responses compared to the RAG Baseline.
- **Lower Hallucination Rate**: While retrieval expansion often increases hallucination, our model maintains lower hallucination levels than the RAG Baseline.
- **Higher Exact Match Accuracy**: Nearly **10× improvement** over the Vanilla Baseline in **exactly correct** responses.

---

---

## **Installation and Setup**
Follow these steps to set up the environment and run the pipeline.

### **Create Conda Environment**
```sh
conda create -n rag python=3.10
conda activate rag
```

### **Install Dependencies**
```sh
pip install -r requirements.txt
pip install --upgrade openai
```

### **Authenticate Hugging Face**
1. Create an account at [Hugging Face](https://huggingface.co/)
2. Generate an access token
3. Log in using:
```sh
huggingface-cli login --token "your_access_token"
```

### **Start vLLM Server**
To enable efficient LLM inference, start the **vLLM server**:
```sh
export CUDA_VISIBLE_DEVICES=0
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --gpu_memory_utilization=0.85 \
  --tensor_parallel_size=1 \
  --dtype="half" \
  --port=8088 \
  --enforce_eager
```

---

## **Running the Model**

### **Generate Model Responses**
#### **With vLLM Server**
```sh
python generate.py \
  --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
  --split 1 \
  --model_name "rag_updated" \
  --llm_name "meta-llama/Llama-3.2-3B-Instruct" \
  --is_server \
  --vllm_server "http://localhost:8088/v1"
```
#### **Without vLLM Server (Offline Inference)**
```sh
python generate.py \
  --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
  --split 1 \
  --model_name "rag_updated" \
  --llm_name "meta-llama/Llama-3.2-3B-Instruct"
```

---

## **Evaluating Model Performance**

### **Evaluate Generated Responses**
#### **With vLLM Server**
```sh
python evaluation_script.py \
  --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
  --model_name "rag_updated" \
  --llm_name "meta-llama/Llama-3.2-3B-Instruct" \
  --is_server \
  --vllm_server "http://localhost:8088/v1" \
  --max_retries 10
```
#### **Without vLLM Server**
```sh
python evaluation_script.py \
  --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
  --model_name "rag_updated" \
  --llm_name "meta-llama/Llama-3.2-3B-Instruct" \
  --max_retries 10
```
