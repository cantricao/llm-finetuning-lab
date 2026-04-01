import logging
import re
import pandas as pd
from google import genai
from google.genai import types

from src.core.config import SyntheticDataConfig

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Handles generation of synthetic training data using Google Gemini API.
    """

    def __init__(self, config: SyntheticDataConfig) -> None:
        """
        Initializes the generator with the provided configuration.
        
        Args:
            config (SyntheticDataConfig): Validated configuration including API keys.
        """
        self.config = config
        self.client = genai.Client(api_key=self.config.api_key.get_secret_value())
        logger.info("Initialized Google GenAI Client.")

    @staticmethod
    def remove_irrelevant_sections(description: str) -> str:
        """
        Cleans job descriptions by removing EOE and standard boilerplate text.
        
        Args:
            description (str): Raw job description.
            
        Returns:
            str: Cleaned job description.
        """
        if not isinstance(description, str):
            return ""
            
        patterns = [
            r"EOE.*?(?=\n|$)",
            r"EEO.*?(?=\n|$)",
            r"equal employment*?(?=\n|$)",
            r"Equal employment opportunity.*?(?=\n|$)"
        ]
        
        for pattern in patterns:
            description = re.sub(pattern, "", description, flags=re.IGNORECASE | re.DOTALL)
            
        return description.strip()

    def upload_batch_requests(self) -> str:
        """
        Uploads the JSONL batch file to Google GenAI for processing.
        
        Returns:
            str: The ID/Name of the uploaded batch file.
        """
        logger.info(f"Uploading batch file from {self.config.batch_file_path}")
        try:
            batch_input_file = self.client.files.upload(
                file=self.config.batch_file_path,
                config=types.UploadFileConfig(
                    display_name='synthetic-batch-requests',
                    mime_type='application/jsonl'
                )
            )
            logger.info(f"Successfully uploaded file: {batch_input_file.name}")
            return batch_input_file.name
        except Exception as e:
            logger.error(f"Failed to upload batch file: {e}")
            raise RuntimeError(f"Batch upload error: {e}") from e
