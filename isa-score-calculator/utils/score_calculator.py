import pandas as pd
import numpy as np
from config import config
from collections import Counter

def calculate_isa_scores(judge_scores_df: pd.DataFrame) -> dict:
    """
    Calculate final ISA scores with majority voting
    
    Returns dict with:
    - overall_score: float
    - category_scores: dict {category: score}
    - topic_scores: list of scores per topic
    - detailed_scores: DataFrame with all calculations
    """
    
    # Create results dataframe
    results_df = judge_scores_df[['Criterion ID', 'Criterion']].copy()
    
    # Step 1: Calculate final score for each scenario using majority vote
    for scenario_num in range(1, 6):
        final_scores = []
        
        for idx, row in judge_scores_df.iterrows():
            # Get the 3 judge scores for this scenario
            judge_scores = [
                row[f'S{scenario_num}_J1'],
                row[f'S{scenario_num}_J2'],
                row[f'S{scenario_num}_J3']
            ]
            
            # Remove None values
            judge_scores = [s for s in judge_scores if s is not None]
            
            if not judge_scores:
                final_scores.append(None)
                continue
            
            # Majority vote
            final_score = majority_vote(judge_scores)
            final_scores.append(final_score)
        
        results_df[f'Final_S{scenario_num}'] = final_scores
    
    # Step 2: Calculate average score for each topic (average of 5 scenarios)
    topic_scores = []
    for idx, row in results_df.iterrows():
        scenario_scores = [
            row[f'Final_S{i}'] 
            for i in range(1, 6) 
            if row[f'Final_S{i}'] is not None
        ]
        
        if scenario_scores:
            topic_score = np.mean(scenario_scores)
            topic_scores.append(topic_score)
        else:
            topic_scores.append(None)
    
    results_df['Topic_Score'] = topic_scores
    
    # Step 3: Calculate category scores
    category_scores = {}
    for category, (start, end) in zip(config.CATEGORIES, config.CATEGORY_RANGES):
        category_topic_scores = [
            score for score in topic_scores[start:end] 
            if score is not None
        ]
        
        if category_topic_scores:
            category_scores[category] = np.mean(category_topic_scores)
        else:
            category_scores[category] = None
    
    # Step 4: Calculate overall score
    valid_topic_scores = [s for s in topic_scores if s is not None]
    overall_score = np.mean(valid_topic_scores) if valid_topic_scores else None
    
    return {
        'overall_score': overall_score,
        'category_scores': category_scores,
        'topic_scores': topic_scores,
        'detailed_scores': results_df
    }

def majority_vote(scores: list) -> float:
    """
    Get majority vote from list of scores
    If no majority, return the average
    """
    if not scores:
        return None
    
    # Count occurrences
    counter = Counter(scores)
    most_common = counter.most_common(1)[0]
    
    # Check if there's a clear majority (more than half)
    if most_common[1] > len(scores) / 2:
        return most_common[0]
    
    # If no clear majority, return average
    return np.mean(scores)