import os
import anthropic
from dotenv import load_dotenv


class ModelClaude:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in the .env file.")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = "claude-sonnet-4-6"

    async def gemini_response(self, prompt: str) -> str:
        """Drop-in replacement for ModelGemini.gemini_response using Claude."""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"[Claude API Error: {str(e)}]"
