"""
OpenRouter client for the LLM calls of the Path Explorer and the Feature Selector: JSON replies and token costs.
"""
import os
import json
import requests
import re
from typing import Dict, List
import logging

from src.cost_logger import cost_logger

COMPONENT_MAP = {
    'path_explorer': 'PEX',
    'feature_selector': 'FS',
    'default': 'OTHER',
}


# Reasoning models (e.g. gpt-6-luna) spend completion tokens on hidden reasoning before the visible answer. Every
# request gets this headroom on top of the caller's budget, so the answer is not cut off. The cap is an upper
# bound, so models that do not reason are unaffected.
REASONING_HEADROOM = 4000


class LLMResponseParseError(RuntimeError):
    """The LLM answered, but not with the JSON the caller asked for (API failures stay plain RuntimeErrors)."""


class HippasusLLMClient:

    def __init__(self, component: str = "default"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")

        # The Path Explorer and the Feature Selector can each use their own model
        if component == "path_explorer":
            self.model = os.environ.get("HIPPASUS_PATH_EXPLORER_LLM") or os.environ.get("HIPPASUS_LLM_MODEL")
        elif component == "feature_selector":
            self.model = os.environ.get("HIPPASUS_FEATURE_SELECTOR_LLM") or os.environ.get("HIPPASUS_LLM_MODEL")
        else:
            self.model = os.environ.get("HIPPASUS_LLM_MODEL")

        if not self.model:
            raise ValueError("HIPPASUS_LLM_MODEL environment variable required")

        self.component = component
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"HippasusLLMClient[{component}] initialized with model: {self.model}")

    def call_for_json_object(self, prompt: str, max_tokens: int = 1200,
                             timeout: int = 60) -> Dict[str, float]:
        content = self._make_api_call(prompt, max_tokens, timeout)

        print(" LLM RESPONSE (JSON Object):")
        print(content)
        print("=" * 60)

        return self._parse_json_object(content)

    def call_for_json_array(self, system_prompt: str, user_prompt: str,
                            max_tokens: int = 2000, timeout: int = 60) -> List[str]:
        content = self._make_api_call_with_system(system_prompt, user_prompt, max_tokens, timeout)

        print(" LLM RESPONSE (JSON Array):")
        print(content)
        print("=" * 60)

        return self._parse_json_array(content)

    def _visible_content(self, response_json: dict) -> str:
        """The reply's text, after logging its usage. Empty or cut-off replies are unusable."""
        if 'usage' in response_json:
            cost_logger.log_call(component=COMPONENT_MAP.get(self.component, self.component.upper()),
                                 model=self.model, usage=response_json['usage'], call_type="always")
        choice = response_json["choices"][0]
        content = choice["message"].get("content") or ""
        if choice.get("finish_reason") == "length":
            raise LLMResponseParseError("LLM reply cut off at the token limit")
        if not content.strip():
            raise LLMResponseParseError("Empty LLM reply")
        return content.strip()

    def _make_api_call(self, prompt: str, max_tokens: int, timeout: int) -> str:
        """Make API call with single user message."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hippasus.local",
            "X-Title": "Hippasus Research"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens + REASONING_HEADROOM
        }

        cost_logger.start_timer()
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

        try:
            response_json = response.json()

            if "error" in response_json:
                error_msg = response_json.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"LLM API error: {error_msg}")

            if "choices" not in response_json:
                raise RuntimeError(f"Invalid API response: {response_json}")

            return self._visible_content(response_json)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid LLM API response format: {e}")

    def _make_api_call_with_system(self, system_prompt: str, user_prompt: str,
                                   max_tokens: int, timeout: int) -> str:
        """Make API call with system and user messages."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hippasus.local",
            "X-Title": "Hippasus Research"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens + REASONING_HEADROOM
        }

        cost_logger.start_timer()
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

        try:
            response_json = response.json()

            if "error" in response_json:
                error_msg = response_json.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"LLM API error: {error_msg}")

            if "choices" not in response_json:
                raise RuntimeError(f"Invalid API response: {response_json}")

            return self._visible_content(response_json)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid LLM API response format: {e}")

    def _parse_json_object(self, content: str) -> Dict[str, float]:
        """
        Parse LLM response expecting JSON object with NUMERIC values.

        Used by Path Explorer for table scoring.
        Validates that all values are numbers and converts to floats in [0, 1].
        """
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise LLMResponseParseError("No JSON object found in LLM response")

        try:
            parsed_obj = json.loads(json_match.group())
            if not isinstance(parsed_obj, dict):
                raise LLMResponseParseError("Response is not a JSON object")

            validated_scores = {}
            for key, value in parsed_obj.items():
                try:
                    validated_scores[str(key)] = max(0.0, min(1.0, float(value)))
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid score for {key}: {value}, using 0.5")
                    validated_scores[str(key)] = 0.5

            return validated_scores

        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"Invalid JSON in LLM response: {e}")

    def _parse_json_array(self, content: str) -> List[str]:
        """Parse LLM response expecting JSON array."""
        # Handle markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                if part.strip().startswith("json"):
                    content = part[4:].strip()
                    break
                elif "[" in part and "]" in part:
                    content = part.strip()
                    break

        start_idx = content.find("[")
        end_idx = content.rfind("]") + 1
        if start_idx == -1 or end_idx == 0:
            raise LLMResponseParseError("No JSON array found in LLM response")

        try:
            json_str = content[start_idx:end_idx]
            parsed_array = json.loads(json_str)

            if not isinstance(parsed_array, list):
                raise LLMResponseParseError("Response is not a JSON array")

            return [str(item).strip() for item in parsed_array]

        except json.JSONDecodeError as e:
            raise LLMResponseParseError(f"Invalid JSON array in LLM response: {e}")
