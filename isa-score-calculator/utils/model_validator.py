from typing import List

from .openrouter_client import OpenRouterClient


def validate_models(api_key: str, contester_model: str, judge_models: List[str]) -> None:
    """
    Ensure both the contester model and all judge models can return responses.
    
    Raises:
        Exception: if any model fails the connectivity test.
    """
    sanitized_judges = [model.strip() for model in judge_models if model and model.strip()]
    if not sanitized_judges:
        raise ValueError("At least one judge model must be provided for evaluation.")
    
    client = OpenRouterClient(api_key)
    
    # Validate contester
    client.validate_model(contester_model, is_judge=False)
    
    # Validate judges
    for judge_model in sanitized_judges:
        client.validate_model(judge_model, is_judge=True)

