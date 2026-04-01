from pydantic import BaseModel, Field
from typing import Optional

class TrainingConfig(BaseModel):
    """
    Configuration parameters for LLM fine-tuning.
    Ensures type safety and parameter validation before training starts.
    """
    model_name: str = Field(..., description="HuggingFace model ID to finetune.")
    dataset_path: str = Field(..., description="Path or HuggingFace ID of the dataset.")
    batch_size: int = Field(default=4, ge=1, description="Training batch size.")
    learning_rate: float = Field(default=2e-4, gt=0.0, description="Optimizer learning rate.")
    max_steps: int = Field(default=1000, ge=10, description="Maximum training steps.")
    lora_r: int = Field(default=16, description="LoRA attention dimension.")
    lora_alpha: int = Field(default=32, description="LoRA alpha parameter.")
    output_dir: str = Field(default="./outputs", description="Directory to save checkpoints.")
    use_4bit: bool = Field(default=True, description="Enable 4-bit quantization.")

    class Config:
        frozen = True # Prevent accidental modification during runtime
