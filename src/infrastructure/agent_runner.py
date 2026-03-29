"""Agent 执行框架。

封装 LLM API 调用，支持多模型后端（Anthropic / OpenAI / OpenAI-Compatible），
提供并发控制、重试、token 计数等功能。
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """单次 Agent 执行结果。"""
    run_id: str
    task_id: str
    model: str
    skill_config: dict
    output: str
    tokens_used: int
    latency_ms: float
    raw_response: dict = field(default_factory=dict)


class AgentRunner:
    """Agent 执行器，管理多模型 API 调用与并发。"""

    def __init__(self, config: dict):
        self.config = config
        self.models = {m["name"]: m for m in config["agents"]["models"]}
        self.max_concurrent = config["agents"].get("concurrency", 20)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._clients: dict = {}

    def _get_client(self, model_name: str):
        """延迟初始化 API client。"""
        if model_name in self._clients:
            return self._clients[model_name]

        model_cfg = self.models[model_name]
        provider = model_cfg["provider"]

        if provider == "anthropic":
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()
        elif provider in ("openai", "openai_compatible"):
            from openai import AsyncOpenAI
            kwargs = {}
            if "base_url" in model_cfg:
                kwargs["base_url"] = model_cfg["base_url"]
            client = AsyncOpenAI(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._clients[model_name] = (client, provider)
        return client, provider

    async def run(
        self,
        run_id: str,
        task_id: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AgentResponse:
        """执行单次 Agent 调用。"""
        async with self._semaphore:
            return await self._run_with_retry(
                run_id, task_id, model_name, system_prompt, user_prompt
            )

    async def _run_with_retry(
        self,
        run_id: str,
        task_id: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        max_retries: Optional[int] = None,
    ) -> AgentResponse:
        retry_cfg = self.config["agents"].get("retry", {})
        max_retries = max_retries or retry_cfg.get("max_retries", 3)
        backoff_base = retry_cfg.get("backoff_base", 2.0)

        for attempt in range(max_retries + 1):
            try:
                start = time.monotonic()
                output, tokens, raw = await self._call_api(
                    model_name, system_prompt, user_prompt
                )
                latency = (time.monotonic() - start) * 1000

                return AgentResponse(
                    run_id=run_id,
                    task_id=task_id,
                    model=model_name,
                    skill_config={},
                    output=output,
                    tokens_used=tokens,
                    latency_ms=latency,
                    raw_response=raw,
                )
            except Exception as e:
                if attempt == max_retries:
                    raise
                wait = backoff_base ** attempt
                logger.warning(
                    f"Run {run_id} attempt {attempt+1} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)

    async def _call_api(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, dict]:
        """调用 LLM API，返回 (output, tokens, raw_response)。"""
        client, provider = self._get_client(model_name)
        model_cfg = self.models[model_name]

        if provider == "anthropic":
            response = await client.messages.create(
                model=model_cfg["model_id"],
                max_tokens=model_cfg.get("max_tokens", 4096),
                temperature=model_cfg.get("temperature", 0),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            output = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            raw = json.loads(response.model_dump_json())
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await client.chat.completions.create(
                model=model_cfg["model_id"],
                messages=messages,
                max_tokens=model_cfg.get("max_tokens", 4096),
                temperature=model_cfg.get("temperature", 0),
            )
            output = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            raw = response.model_dump()

        return output, tokens, raw

    async def run_batch(
        self,
        runs: list[dict],
    ) -> list[AgentResponse]:
        """批量执行多个 Agent 调用，自动并发控制。"""
        tasks = [
            self.run(
                run_id=r["run_id"],
                task_id=r["task_id"],
                model_name=r["model"],
                system_prompt=r["system_prompt"],
                user_prompt=r["user_prompt"],
            )
            for r in runs
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
