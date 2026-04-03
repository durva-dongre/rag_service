# rag_service/rag_service/core/llm_providers.py

import aiohttp
import asyncio
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .llm_interface import BaseLLMInterface


class OpenAIProvider(BaseLLMInterface):
    """OpenAI provider using LangChain"""

    def __init__(self, api_key: str, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        super().__init__(api_key, model_name, temperature, max_tokens)
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )

    async def generate(self, messages: List[Dict]) -> str:
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            else:
                langchain_messages.append(HumanMessage(content=msg["content"]))

        response = await self.llm.agenerate([langchain_messages])
        return response.generations[0][0].text.strip()

    async def generate_with_vision(self, messages: List[Dict]) -> str:
        return await self.generate(messages)


class GroqProvider(BaseLLMInterface):
    """
    Groq provider — uses Groq's OpenAI-compatible REST API directly.

    Recommended free vision model:
      meta-llama/llama-4-scout-17b-16e-instruct   (Llama 4 Scout, vision capable)

    Other free Groq models (text only):
      llama3-8b-8192
      llama3-70b-8192
      mixtral-8x7b-32768
      gemma2-9b-it

    In Frappe LLM Settings:
      Provider  : groq
      API Key   : <your Groq API key>   (put it where OpenAI key was)
      Model     : meta-llama/llama-4-scout-17b-16e-instruct
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        super().__init__(api_key, model_name, temperature, max_tokens)
        self.timeout = aiohttp.ClientTimeout(total=60)

    def _build_headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: List[Dict]) -> Dict:
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    async def generate(self, messages: List[Dict]) -> str:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                self.BASE_URL,
                headers=self._build_headers(),
                json=self._build_payload(messages),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API error {response.status}: {error_text}")
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()

    async def generate_with_vision(self, messages: List[Dict]) -> str:
        """
        For vision, pass messages with content as a list containing
        text + image_url blocks (same format as OpenAI vision).

        Example message format:
          {
            "role": "user",
            "content": [
              {"type": "text", "text": "What is in this image?"},
              {"type": "image_url", "image_url": {"url": "https://..."}}
            ]
          }
        """
        return await self.generate(messages)

    def format_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_url: Optional[str] = None,
    ) -> List[Dict]:
        """Helper to build a properly formatted message list for Groq/Llama vision."""
        if image_url:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class TogetherAIProvider(BaseLLMInterface):
    """Together AI provider optimized for Llama 3.2 90B Vision"""

    def __init__(self, api_key: str, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        super().__init__(api_key, model_name, temperature, max_tokens)
        self.base_url = "https://api.together.xyz/v1/chat/completions"

        self.is_llama_32_90b = "Llama-3.2-90B" in model_name
        if self.is_llama_32_90b:
            self.timeout = 90
            self.max_retries = 3

    async def generate(self, messages: List[Dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "stream": False,
        }

        if self.is_llama_32_90b:
            data.update({
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": ["</s>", "<|eot_id|>"],
            })

        timeout = aiohttp.ClientTimeout(total=self.timeout if hasattr(self, "timeout") else 60)
        max_retries = self.max_retries if hasattr(self, "max_retries") else 1

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(max_retries):
                try:
                    async with session.post(self.base_url, headers=headers, json=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            raise Exception(f"Together AI API error: {response.status} - {error_text}")
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise Exception("Request timeout — Llama 3.2 90B may need more processing time")

    async def generate_with_vision(self, messages: List[Dict]) -> str:
        return await self.generate(messages)

    def format_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_url: Optional[str] = None,
    ) -> List[Dict]:
        messages = []
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        if image_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                ],
            })
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        return messages


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_provider(
    provider: str,
    api_key: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> BaseLLMInterface:
    """Factory function to create LLM provider instances.

    Supported provider strings (case-insensitive match on lower):
      openai        → OpenAIProvider
      groq          → GroqProvider
      together ai   → TogetherAIProvider
    """

    # Normalise so Frappe UI values like "groq" / "Groq" both work
    _key = provider.strip().lower()

    _map = {
        "openai":      OpenAIProvider,
        "groq":        GroqProvider,
        "together ai": TogetherAIProvider,
        "togetherai":  TogetherAIProvider,
    }

    # Also keep original-case lookup for backwards compat ("OpenAI", "Together AI")
    _map_orig = {
        "OpenAI":      OpenAIProvider,
        "Groq":        GroqProvider,
        "Together AI": TogetherAIProvider,
    }

    cls = _map.get(_key) or _map_orig.get(provider)

    if cls is None:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported values: openai, groq, together ai"
        )

    return cls(api_key, model_name, temperature, max_tokens)
