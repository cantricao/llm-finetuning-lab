import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import TrainingConfig
from src.models.unsloth_trainer import UnslothTrainer
from src.core.logger import get_logger

logger = get_logger(__name__)

def main() -> None:
    """
    Entry point for the model fine-tuning pipeline.
    """
    logger.info("Bootstrapping Unsloth Training Pipeline...")
    
    config = TrainingConfig(
        model_id="unsloth/llama-3-8b-bnb-4bit",
        lora_r=16
    )
    
    trainer = UnslothTrainer(config=config)
    
    try:
        trainer.prepare_model()
        logger.info("Model prepared successfully. Ready to ingest datasets.")
        # dataset = load_dataset(...)
        # trainer.train(dataset)
    except Exception as e:
        logger.critical(f"Training pipeline crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
