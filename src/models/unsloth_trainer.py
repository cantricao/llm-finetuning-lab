import logging
from typing import Any
from unsloth import FastVisionModel

from src.core.config import TrainingConfig

logger = logging.getLogger(__name__)

class UnslothTrainer:
    """
    Encapsulates the fine-tuning logic for Vision/Language Models using Unsloth.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initializes the training pipeline.
        
        Args:
            config (TrainingConfig): Validated training parameters.
        """
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        logger.info(f"Initialized UnslothTrainer for model: {self.config.model_id}")

    def prepare_model(self) -> None:
        """
        Downloads and prepares the base model with quantization and LoRA adapters.
        Raises RuntimeError if the initialization fails.
        """
        logger.info("Starting model preparation phase...")
        try:
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=self.config.model_id,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
            )
            
            logger.info("Applying LoRA adapters...")
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0,
                bias="none",
                use_rslora=False,
            )
            logger.info("Model successfully injected with PEFT/LoRA modules.")
            
        except Exception as e:
            logger.error(f"Failed to prepare Unsloth model: {e}")
            raise RuntimeError(f"Model initialization error: {e}") from e
