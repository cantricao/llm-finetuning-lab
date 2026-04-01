import logging
from typing import Any, Callable, Dict, List
from ollama import Client

from src.core.config import AgentConfig

logger = logging.getLogger(__name__)

class OllamaFunctionCaller:
    """
    Manages autonomous function calling using local Ollama models.
    """

    def __init__(self, config: AgentConfig, available_tools: Dict[str, Callable]) -> None:
        """
        Initializes the function calling agent.
        
        Args:
            config (AgentConfig): Configuration for the Ollama client.
            available_tools (Dict[str, Callable]): Mapping of tool names to Python functions.
        """
        self.config = config
        self.client = Client(host=self.config.host)
        self.available_tools = available_tools
        logger.info(f"Initialized OllamaFunctionCaller with model: {self.config.model_name}")

    def run_inference(self, prompt: str, tools_schema: List[Dict[str, Any]]) -> str:
        """
        Runs the LLM, parses tool calls, executes them locally, and returns the final answer.
        
        Args:
            prompt (str): The user's input prompt.
            tools_schema (List[Dict[str, Any]]): The JSON schema defining available tools.
            
        Returns:
            str: The final text response from the LLM.
        """
        messages = [
            {
                "role": "system",
                "content": "Based on the information returned by function calling, answer the question politely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        logger.info("Sending prompt to local LLM...")
        
        try:
            response = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                tools=tools_schema
            )
            
            response_message = response.message
            
            if not response_message.tool_calls:
                logger.info("No tool calls triggered by the LLM.")
                return response_message.content

            messages.append(response_message)
            
            # Safely execute tools requested by the LLM
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments
                
                logger.info(f"Executing local function: {func_name} with args: {func_args}")
                if func_name not in self.available_tools:
                    raise ValueError(f"Tool {func_name} is not registered in available_tools.")
                    
                func_to_call = self.available_tools[func_name]
                result = func_to_call(**func_args)
                
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "name": func_name
                })
                
            logger.info("Sending tool results back to LLM for final synthesis...")
            final_response = self.client.chat(
                model=self.config.model_name,
                messages=messages
            )
            return final_response.message.content
            
        except Exception as e:
            logger.error(f"Agent inference failed: {e}")
            raise RuntimeError(f"Failed to execute function calling: {e}") from e
