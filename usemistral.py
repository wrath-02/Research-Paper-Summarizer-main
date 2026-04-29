import os
import sys
import httpx
from dotenv import load_dotenv

QUOTA_KEYWORDS = ["429", "quota", "rate_limit", "too_many_requests", "exhausted"]
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralQuotaError(Exception):
    pass


class ModelMistral:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            print("ERROR: MISTRAL_API_KEY is not set in the .env file.", file=sys.stderr)
            raise ValueError("MISTRAL_API_KEY not set.")
        self.api_key = api_key
        self.model = "mistral-large-latest"

    async def gemini_response(self, prompt: str) -> str:
        """Drop-in replacement for ModelGemini.gemini_response using Mistral REST API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(MISTRAL_API_URL, headers=headers, json=payload)
                if response.status_code == 429:
                    print(
                        "\n❌ MISTRAL API RATE LIMIT HIT\n"
                        "Options:\n"
                        "  1. Wait and retry later\n"
                        "  2. Upgrade your Mistral plan at https://console.mistral.ai\n",
                        file=sys.stderr,
                    )
                    raise MistralQuotaError(f"HTTP 429: {response.text}")
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except MistralQuotaError:
            raise
        except Exception as e:
            err = str(e)
            if any(k in err.lower() for k in QUOTA_KEYWORDS):
                raise MistralQuotaError(err)
            return f"[Mistral Error: {err}]"
