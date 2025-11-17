import pandas as pd
import time
from .openrouter_client import OpenRouterClient
from config import config

def get_judge_scores(
    api_key: str,
    answers_df: pd.DataFrame,
    judge_models=None,
    progress_callback=None
) -> pd.DataFrame:
    """
    Get judge scores for all 150 answers
    
    Returns DataFrame with 30 rows and 15 columns (5 scenarios * N judges):
    - S1_J1, S1_J2, ..., S5_JN
    """
    judge_models = judge_models or config.JUDGE_MODELS
    
    if not judge_models:
        raise ValueError("At least one judge model must be provided.")
    
    # Initialize client
    client = OpenRouterClient(api_key)
    
    # Create result dataframe
    result_df = pd.DataFrame()
    result_df['Criterion ID'] = answers_df['Criterion ID']
    result_df['Criterion'] = answers_df['Criterion']
    
    # Total evaluations: 30 topics * 5 scenarios * number of judges
    total_evaluations = 30 * 5 * len(judge_models)
    current_evaluation = 0
    
    # For each scenario (1-5)
    for scenario_num in range(1, 6):
        # For each judge
        for judge_idx, judge_model in enumerate(judge_models, 1):
            scores = []
            
            # For each topic (row)
            for idx, row in answers_df.iterrows():
                criterion = row['Criterion']
                scenario = row[f'Scenario_{scenario_num}']
                answer = row[f'Answer_{scenario_num}']
                
                # Update progress
                current_evaluation += 1
                if progress_callback:
                    progress_callback(
                        current_evaluation,
                        total_evaluations,
                        f"Judge {judge_idx} evaluating Topic {idx+1}, Scenario {scenario_num}"
                    )
                
                # Skip if no answer
                if pd.isna(answer) or answer == "" or str(answer).startswith("ERROR"):
                    scores.append(None)
                    continue
                
                try:
                    # Get judge's score
                    score = client.get_judge_score(criterion, scenario, answer, judge_model)
                    scores.append(score)
                    
                    # Delay between requests
                    time.sleep(config.DELAY_BETWEEN_REQUESTS)
                    
                except Exception as e:
                    print(f"Error getting score from {judge_model} for row {idx}, scenario {scenario_num}: {str(e)}")
                    scores.append(None)
            
            # Add scores column with clear naming: S1_J1, S1_J2, S1_J3, etc.
            result_df[f'S{scenario_num}_J{judge_idx}'] = scores
    
    return result_df