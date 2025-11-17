from openai import OpenAI
from config import config
import re

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=api_key,
        )
    
    def get_model_answer(self, scenario: str, model_name: str) -> str:
        """Get model's answer to a scenario (no system prompt, very low temperature)"""
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": scenario}],
                temperature=config.MODEL_TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling model {model_name}: {str(e)}")
    
    def validate_model(self, model_name: str, is_judge: bool = False) -> None:
        """Send a lightweight ping request to ensure the model is reachable."""
        messages = [{"role": "user", "content": config.MODEL_VALIDATION_PROMPT}]
        if is_judge:
            messages.insert(0, {"role": "system", "content": config.JUDGE_SYSTEM_PROMPT})
        
        temperature = config.JUDGE_TEMPERATURE if is_judge else config.MODEL_TEMPERATURE
        
        try:
            self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=5
            )
        except Exception as e:
            role = "judge" if is_judge else "contester"
            raise Exception(f"{role.title()} model validation failed ({model_name}): {str(e)}")
    
    def get_judge_score(self, criterion: str, scenario: str, answer: str, judge_model: str) -> int:
        """Get judge's score for a model's answer"""
        try:
            # Create the prompt for the judge
            user_prompt = f"""
Criterion: {criterion}

Scenario: {scenario}

Expert's Answer: {answer}

Please rate this answer with a score of 1, 2, or 3 according to the guidelines provided. Respond with ONLY the number.
"""
            
            response = self.client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": config.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=10
            )
            
            # Extract the score (1, 2, or 3)
            response_text = response.choices[0].message.content.strip()
            score = self._extract_score(response_text)
            
            return score
            
        except Exception as e:
            raise Exception(f"Error calling judge {judge_model}: {str(e)}")
    
    def _extract_score(self, text: str) -> int:
        """Extract score from judge's response"""
        # Look for a number 1, 2, or 3
        match = re.search(r'\b([123])\b', text)
        if match:
            return int(match.group(1))
        
        # If no clear number, try to parse the first character
        if text and text[0] in ['1', '2', '3']:
            return int(text[0])
        
        # Default to 2 if unclear
        print(f"Warning: Could not extract clear score from: {text}. Defaulting to 2.")
        return 2