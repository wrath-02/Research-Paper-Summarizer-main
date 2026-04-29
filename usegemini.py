import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

QUOTA_KEYWORDS = ["429", "quota", "RESOURCE_EXHAUSTED", "exhausted", "rateLimitExceeded"]


class GeminiQuotaError(Exception):
    pass


class ModelGemini:
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("ERROR: GEMINI_API_KEY is not set in the .env file.", file=sys.stderr)
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    async def gemini_response(self, prompt: str) -> str:
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            err = str(e)
            if any(k in err for k in QUOTA_KEYWORDS):
                print(
                    "\n❌ GEMINI API QUOTA EXHAUSTED\n"
                    "Your Gemini API key has run out of quota.\n"
                    "Fix options:\n"
                    "  1. Wait and try again later (free tier resets daily)\n"
                    "  2. Get a new API key from https://aistudio.google.com/app/apikey\n"
                    "  3. Add billing to your Google Cloud project\n",
                    file=sys.stderr,
                )
                raise GeminiQuotaError(err)
            return f"[Gemini Error: {err}]"
