import os
from typing import Optional, Tuple, Union

import httpx
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from deepeval.models import AnthropicModel, GPTModel, GeminiModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from network_env import disable_broken_loopback_proxy_env


def _clean_env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()


class XAICompatibleJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        super().__init__(model)

    def load_model(self, async_mode: bool = False):
        disable_broken_loopback_proxy_env()
        if async_mode:
            return AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.AsyncClient(trust_env=False),
            )
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(trust_env=False),
        )

    def _schema_prompt(self, prompt: str, schema: BaseModel) -> str:
        schema_json = schema.model_json_schema()
        return (
            f"{prompt}\n\n"
            "Return only valid JSON matching this schema exactly.\n"
            f"Schema: {schema_json}"
        )

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=False)
        user_prompt = self._schema_prompt(prompt, schema) if schema else prompt
        completion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        output = (completion.choices[0].message.content or "").strip()
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), 0.0
        return output, 0.0

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)
        user_prompt = self._schema_prompt(prompt, schema) if schema else prompt
        completion = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        output = (completion.choices[0].message.content or "").strip()
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), 0.0
        return output, 0.0

    def get_model_name(self, *args, **kwargs) -> str:
        return self.name


def build_judge_model(*, purpose: str = "metrics", for_simulator: bool = False):
    """
    Build the DeepEval judge model from environment variables.

    Supported providers:
    - google / gemini
    - xai / grok
    - openai
    - anthropic / claude

    Useful env vars:
    - DEEPEVAL_JUDGE_PROVIDER
    - DEEPEVAL_JUDGE_MODEL
    - GOOGLE_API_KEY
    - XAI_API_KEY
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    """
    provider = _clean_env(
        "DEEPEVAL_JUDGE_PROVIDER",
        _clean_env("CHEEKO_PROVIDER", "google"),
    ).lower()
    model_name = _clean_env("DEEPEVAL_JUDGE_MODEL")

    if provider in ("google", "gemini"):
        api_key = _clean_env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing. Set DEEPEVAL_JUDGE_PROVIDER to xai, "
                "openai, or anthropic if Google is blocked."
            )
        kwargs = {}
        if for_simulator:
            # Gemini thinking can break structured schema parsing in the simulator.
            kwargs["generation_kwargs"] = {"thinking_config": {"thinking_budget": 0}}
        return GeminiModel(model_name or "gemini-2.5-flash", api_key=api_key, **kwargs)

    if provider in ("xai", "grok"):
        api_key = _clean_env("XAI_API_KEY")
        base_url = _clean_env("XAI_BASE_URL", "https://api.x.ai/v1")
        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY is missing. Add it to .env to use Grok as the evaluator."
            )
        return XAICompatibleJudgeModel(
            model_name or _clean_env("XAI_MODEL", "grok-4-fast-non-reasoning"),
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "openai":
        api_key = _clean_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to .env to use OpenAI as the evaluator."
            )
        return GPTModel(model_name or "gpt-4.1-mini", api_key=api_key)

    if provider in ("anthropic", "claude"):
        api_key = _clean_env("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing. Add it to .env to use Claude as the evaluator."
            )
        return AnthropicModel(
            model_name or "claude-sonnet-4-20250514",
            api_key=api_key,
        )

    raise RuntimeError(
        f"Unsupported DEEPEVAL_JUDGE_PROVIDER='{provider}'. "
        "Use one of: google, xai, openai, anthropic."
    )
