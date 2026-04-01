from pydantic import BaseModel, Field, SecretStr
from typing import Optional

class TrainingConfig(BaseModel):
    """Configuration for Unsloth Vision/Language Model fine-tuning."""
    model_id: str = Field(default="unsloth/llama-3-8b-bnb-4bit", description="HuggingFace model ID")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    load_in_4bit: bool = Field(default=True, description="Enable 4-bit quantization")
    lora_r: int = Field(default=16, description="LoRA rank parameter")
    lora_alpha: int = Field(default=16, description="LoRA alpha parameter")
    output_dir: str = Field(default="./outputs", description="Path to save checkpoints")

class AgentConfig(BaseModel):
    """Configuration for Ollama-based Function Calling Agent."""
    model_name: str = Field(default="gpt-oss:20b", description="Ollama model name")
    host: str = Field(default="http://localhost:11434", description="Ollama host URL")
    
class SyntheticDataConfig(BaseModel):
    """Configuration for Google GenAI Synthetic Data pipeline."""
    api_key: SecretStr = Field(..., description="Google Gemini API Key")
    batch_file_path: str = Field(default="batch_requests.jsonl", description="Path to batch requests")
