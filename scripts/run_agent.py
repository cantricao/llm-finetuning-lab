import sys
import logging
from pathlib import Path

# Ensure the src directory is accessible
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import AgentConfig
from src.agents.function_caller import OllamaFunctionCaller
from src.core.logger import get_logger

logger = get_logger(__name__)

# Dummy tool for demonstration
def get_flight_times(departure: str, arrival: str) -> str:
    """
    Simulates an API call to fetch flight times.
    """
    logger.info(f"Tool called: Fetching flights from {departure} to {arrival}")
    return f"Flights from {departure} to {arrival} leave at 08:00 AM and 14:00 PM."

def main() -> None:
    """
    Entry point to test the Ollama Function Calling Agent.
    """
    logger.info("Initializing Agent System...")
    
    # 1. Define configuration
    config = AgentConfig(model_name="llama3:instruct") # Adjust model name as needed
    
    # 2. Register tools
    available_tools = {
        "get_flight_times": get_flight_times
    }
    
    # 3. Define the tool schema required by Ollama
    tools_schema = [{
        "type": "function",
        "function": {
            "name": "get_flight_times",
            "description": "Get the flight times between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "The departure city (e.g., 'Ho Chi Minh')"
                    },
                    "arrival": {
                        "type": "string",
                        "description": "The arrival city (e.g., 'Brisbane')"
                    }
                },
                "required": ["departure", "arrival"]
            }
        }
    }]
    
    agent = OllamaFunctionCaller(config=config, available_tools=available_tools)
    
    prompt = "What are the flight times from Ho Chi Minh to Brisbane?"
    logger.info(f"User Prompt: {prompt}")
    
    try:
        response = agent.run_inference(prompt=prompt, tools_schema=tools_schema)
        logger.info(f"Final Agent Response: {response}")
    except Exception as e:
        logger.critical(f"Agent execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
