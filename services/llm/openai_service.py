"""
=====================================================
AI Voice Platform v2 - OpenAI LLM Service
=====================================================
GPT-4o with streaming support for real-time conversations
"""

import asyncio
from typing import AsyncIterator, List, Optional
from loguru import logger

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from .llm_base import (
    LLMServiceBase,
    LLMRequest,
    LLMResponse,
    LLMChunk,
    Message,
    LLMRole
)


class OpenAILLM(LLMServiceBase):
    """
    OpenAI GPT-4o LLM Service with streaming

    Features:
    - GPT-4o: Fast, accurate, multi-modal
    - Streaming responses for real-time delivery
    - Function calling support
    - JSON mode
    - Token-efficient
    """

    # Token counts for common models (rough estimates)
    TOKEN_COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},  # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI LLM service

        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            base_url: Optional custom base URL
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is not installed. Install with: pip install openai")

        super().__init__(api_key, model)

        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize async client
        self._client: Optional[AsyncOpenAI] = None
        self._base_url = base_url

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client"""
        if self._client is None:
            kwargs = {"api_key": self.api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        Non-streaming chat completion

        Args:
            request: LLM request

        Returns:
            Complete LLM response
        """
        client = await self._get_client()

        try:
            # Convert messages to OpenAI format
            messages = [msg.to_dict() for msg in request.messages]

            logger.info(f"OpenAI: Sending {len(messages)} messages to {self.model}")

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""

            return LLMResponse(
                content=content,
                role=LLMRole.ASSISTANT,
                finish_reason=choice.finish_reason,
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0,
                metadata={
                    "model": response.model,
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                }
            )

        except Exception as e:
            logger.error(f"OpenAI: Chat error: {e}")
            raise

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """
        Streaming chat completion

        Args:
            request: LLM request

        Yields:
            LLMChunk objects as tokens are generated
        """
        client = await self._get_client()

        try:
            # Convert messages to OpenAI format
            messages = [msg.to_dict() for msg in request.messages]

            logger.info(f"OpenAI: Streaming {len(messages)} messages to {self.model}")

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            content_so_far = ""

            async for chunk in response:
                # Check if this is the final chunk
                if chunk.choices[0].finish_reason is not None:
                    yield LLMChunk(
                        delta="",
                        content=content_so_far,
                        is_final=True,
                        finish_reason=chunk.choices[0].finish_reason
                    )
                    break

                # Extract delta content
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    content_so_far += delta
                    yield LLMChunk(
                        delta=delta,
                        content=content_so_far,
                        is_final=False
                    )

        except Exception as e:
            logger.error(f"OpenAI: Streaming error: {e}")
            raise

    async def chat_with_tools(self, request: LLMRequest) -> LLMResponse:
        """
        Chat completion with function/tool calling support.
        Non-streaming since we need to see the full response to detect tool calls.

        Args:
            request: LLM request with tools defined

        Returns:
            LLMResponse with content and/or tool_calls
        """
        client = await self._get_client()

        try:
            messages = [msg.to_dict() for msg in request.messages]

            # Convert tools to OpenAI format
            openai_tools = []
            for tool in request.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                })

            logger.info(f"OpenAI: Calling {self.model} with {len(messages)} messages and {len(openai_tools)} tools")

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools
                kwargs["tool_choice"] = "auto"

            response = await client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            content = choice.message.content or ""

            # Extract tool calls if present
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
                logger.info(f"OpenAI: Tool calls: {[t['name'] for t in tool_calls]}")

            return LLMResponse(
                content=content,
                role=LLMRole.ASSISTANT,
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls,
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0,
            )

        except Exception as e:
            logger.error(f"OpenAI: chat_with_tools error: {e}")
            raise

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        # Rough estimate: ~4 characters per token for English
        # For production, use tiktoken library
        return len(text) // 4

    async def get_available_functions(self) -> List[dict]:
        """
        Get available functions for function calling

        Returns:
            List of function definitions
        """
        return [
            {
                "name": "book_appointment",
                "description": "Book an appointment with an accountant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_type": {
                            "type": "string",
                            "enum": ["individual", "corporate"],
                            "description": "Type of client"
                        },
                        "accountant": {
                            "type": "string",
                            "enum": ["Hussam Saadaldin", "Rami Kahwaji", "Abdul"],
                            "description": "Preferred accountant"
                        },
                        "date_time": {
                            "type": "string",
                            "description": "Preferred date and time"
                        }
                    },
                    "required": ["client_type"]
                }
            },
            {
                "name": "transfer_to_human",
                "description": "Transfer the call to a human agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for transfer"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_faq_answer",
                "description": "Get answer from FAQ knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "User's question"
                        }
                    },
                    "required": ["question"]
                }
            }
        ]

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD for a request

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if self.model not in self.TOKEN_COSTS:
            return 0.0

        costs = self.TOKEN_COSTS[self.model]
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]

        return input_cost + output_cost


# Factory function
def create_openai_llm(config: dict) -> OpenAILLM:
    """
    Factory function to create OpenAI LLM service from config

    Args:
        config: Configuration dictionary (from Settings)

    Returns:
        Configured OpenAILLM instance
    """
    return OpenAILLM(
        api_key=config.get('openai_api_key'),
        model=config.get('openai_model', 'gpt-4o'),
        temperature=config.get('openai_temperature', 0.7),
        max_tokens=config.get('openai_max_tokens', 150)  # Reduced for voice calls (faster, shorter responses)
    )
