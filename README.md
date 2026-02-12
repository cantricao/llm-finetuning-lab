# 🧬 LLM Fine-tuning & Optimization Lab

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/Unsloth-Fast%20Training-blueviolet)](https://github.com/unslothai/unsloth)

## 📋 Overview
This repository serves as a research lab for **Large Language Model (LLM) Fine-tuning** and **Optimization**. It focuses on adapting open-source models (Llama 3, Gemma) for specific domains using state-of-the-art techniques like **QLoRA** (Quantized Low-Rank Adaptation) and **Unsloth** acceleration.

## 🧪 Experiments & Notebooks

| ID | Notebook | Description | Tech Stack |
| :--- | :--- | :--- | :--- |
| **01** | [Llama 3 Unsloth Finetune](./01_llama3_unsloth_finetune.ipynb) | **2x faster training** of Llama 3 8B using Unsloth & LoRA on limited GPU VRAM. | `Unsloth`, `Llama-3`, `LoRA` |
| **02** | [Gemma Google Finetune](./02_gemma_finetune_google.ipynb) | Fine-tuning Google's Gemma model for specialized instruction following. | `Transformers`, `Gemma`, `PEFT` |
| **03** | [OSS Function Calling](./03_oss_llm_function_calling.ipynb) | Enabling **Function Calling** capabilities in open-source models (turning them into Agents). | `Hermes`, `Tool Use` |

## 🛠️ Key Technologies

* **Unsloth:** For optimized backpropagation and memory reduction.
* **PEFT / LoRA:** Parameter-Efficient Fine-Tuning.
* **BitsAndBytes:** 4-bit Quantization for running large models on consumer GPUs.
* **Hugging Face Ecosystem:** Transformers, TRL (Transformer Reinforcement Learning).

## 🚀 Usage

To run these notebooks, you will need a GPU environment (Google Colab T4 or NVIDIA RTX 3090/4090).

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Setup:**
    Rename `.env.example` to `.env` and add your Hugging Face Token:
    ```env
    HF_TOKEN=hf_your_token_here
    ```

## 👨‍💻 Author

**Tri**
*Master of Data Analytics (Biomedical Specialization)*

Focused on building private, domain-specific AI models for healthcare and enterprise applications.
