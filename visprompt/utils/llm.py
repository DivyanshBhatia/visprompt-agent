"""LLM / VLM client with unified interface, cost tracking, and retry logic."""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Token pricing (USD per 1K tokens, as of early 2025) ──────────────────────
PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "qwen2-vl-72b": {"input": 0.0, "output": 0.0},  # self-hosted
}


@dataclass
class TokenUsage:
    """Tracks token counts and cost for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        prices = PRICING.get(self.model, {"input": 0.01, "output": 0.03})
        return (
            self.input_tokens * prices["input"] / 1000
            + self.output_tokens * prices["output"] / 1000
        )


@dataclass
class CostTracker:
    """Aggregates costs across all LLM calls in a pipeline run."""

    records: list[TokenUsage] = field(default_factory=list)
    agent_costs: dict[str, float] = field(default_factory=dict)

    def record(self, usage: TokenUsage, agent_name: str = "unknown") -> None:
        self.records.append(usage)
        self.agent_costs[agent_name] = self.agent_costs.get(agent_name, 0.0) + usage.cost_usd

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_latency(self) -> float:
        return sum(r.latency_s for r in self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": len(self.records),
            "total_latency_s": round(self.total_latency, 1),
            "per_agent": {k: round(v, 4) for k, v in self.agent_costs.items()},
        }


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic APIs.

    Handles text, vision (image), and structured JSON output.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 3,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cost_tracker = cost_tracker or CostTracker()
        self._client = self._init_client(provider, api_key)

    def _init_client(self, provider: str, api_key: Optional[str]):
        if provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=api_key) if api_key else OpenAI()
            except ImportError:
                raise ImportError("pip install openai")
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=api_key) if api_key else Anthropic()
            except ImportError:
                raise ImportError("pip install anthropic")
        elif provider == "google":
            try:
                import google.generativeai as genai
                if api_key:
                    genai.configure(api_key=api_key)
                return genai
            except ImportError:
                raise ImportError("pip install google-generativeai")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ── Core call ─────────────────────────────────────────────────────────
    def call(
        self,
        prompt: str,
        system: str = "",
        images: Optional[list[str | Path]] = None,
        json_mode: bool = False,
        agent_name: str = "unknown",
    ) -> str:
        """Send a prompt (with optional images) to the LLM and return text.

        Args:
            prompt: User message text.
            system: System prompt.
            images: List of image file paths or base64 strings.
            json_mode: Request JSON-formatted output.
            agent_name: For cost tracking attribution.

        Returns:
            Model response text.
        """
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                if self.provider == "openai":
                    text, usage = self._call_openai(prompt, system, images, json_mode)
                elif self.provider == "anthropic":
                    text, usage = self._call_anthropic(prompt, system, images, json_mode)
                elif self.provider == "google":
                    text, usage = self._call_google(prompt, system, images, json_mode)
                else:
                    raise ValueError(f"Unknown provider {self.provider}")

                usage.latency_s = time.time() - t0
                usage.model = self.model
                self.cost_tracker.record(usage, agent_name)
                return text

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError("LLM call failed after all retries")

    def call_json(
        self,
        prompt: str,
        system: str = "",
        images: Optional[list[str | Path]] = None,
        agent_name: str = "unknown",
    ) -> dict:
        """Call LLM and parse response as JSON."""
        if "json" not in prompt.lower() and "json" not in system.lower():
            prompt += "\n\nRespond ONLY with valid JSON. No markdown fences."
        text = self.call(prompt, system, images, json_mode=True, agent_name=agent_name)
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return json.loads(text)

    # ── Provider-specific implementations ─────────────────────────────────
    def _call_openai(self, prompt, system, images, json_mode):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        content = [{"type": "text", "text": prompt}]
        if images:
            for img in images:
                b64 = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                })
        messages.append({"role": "user", "content": content})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        usage = TokenUsage(
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )
        return resp.choices[0].message.content, usage

    def _call_anthropic(self, prompt, system, images, json_mode):
        content = []
        if images:
            for img in images:
                b64 = self._encode_image(img)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })
        content.append({"type": "text", "text": prompt})

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        usage = TokenUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, usage

    def _call_google(self, prompt, system, images, json_mode):
        import PIL.Image
        model = self._client.GenerativeModel(self.model, system_instruction=system or None)
        parts = []
        if images:
            for img in images:
                path = Path(img) if not isinstance(img, Path) else img
                parts.append(PIL.Image.open(path))
        parts.append(prompt)

        resp = model.generate_content(parts)
        # Google doesn't always expose token counts
        in_tok = getattr(resp.usage_metadata, "prompt_token_count", 0) if hasattr(resp, "usage_metadata") else 0
        out_tok = getattr(resp.usage_metadata, "candidates_token_count", 0) if hasattr(resp, "usage_metadata") else 0
        usage = TokenUsage(input_tokens=in_tok, output_tokens=out_tok)
        return resp.text, usage

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _encode_image(img: str | Path) -> str:
        """Return base64-encoded image string."""
        if isinstance(img, (str, Path)):
            path = Path(img)
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode()
        # Assume already base64
        return str(img)
