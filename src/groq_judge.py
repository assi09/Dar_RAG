"""Shared Groq judge model for DeepEval — import this in any eval script."""
import json
import os
from groq import Groq
from deepeval.models.base_model import DeepEvalBaseLLM

GROQ_MODEL = "llama-3.3-70b-versatile"


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = GROQ_MODEL):
        self._model = model
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None):
        kwargs = {"temperature": 0.0}
        # When Synthesizer passes a schema, use JSON mode for structured output
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = response.choices[0].message.content

        # Parse and validate against schema if provided — return instance directly
        if schema is not None:
            try:
                data = json.loads(text)
                return schema(**data)
            except Exception:
                return text

        return text

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"groq/{self._model}"
