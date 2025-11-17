import pandas as pd
import time
import streamlit as st
from .openrouter_client import OpenRouterClient
from config import config


@st.cache_data(show_spinner=False)
def _load_scenario_data(file_path: str):
    """
    Load and cache the scenario workbook so we don't hit disk on every run.
    Returns the raw dataframe plus a template with pre-populated scenario columns.
    """
    raw_df = pd.read_excel(file_path)
    
    scenario_template = raw_df[['Criterion ID', 'Criterion']].copy()
    for i in range(1, 6):
        column_name = f'Scenario No. {i}'
        scenario_template[f'Scenario_{i}'] = raw_df.get(column_name, "")
    
    return raw_df, scenario_template

def get_model_answers(api_key: str, model_name: str, progress_callback=None) -> pd.DataFrame:
    """
    Get model answers for all 150 scenarios
    
    Returns DataFrame with 30 rows and columns:
    - Criterion ID
    - Criterion
    - Scenario_1, Scenario_2, ..., Scenario_5
    - Answer_1, Answer_2, ..., Answer_5
    """
    # Load cached scenarios
    raw_df, scenario_template = _load_scenario_data(config.SCENARIOS_FILE)
    
    # Initialize client
    client = OpenRouterClient(api_key)
    
    # Create result dataframe
    result_df = scenario_template.copy()
    
    # Get answers for each scenario
    num_topics = len(raw_df.index)
    total_scenarios = 0
    for i in range(1, 6):
        column_name = f'Scenario No. {i}'
        if column_name not in raw_df.columns:
            continue
        total_scenarios += raw_df[column_name].apply(
            lambda val: not (pd.isna(val) or str(val).strip() == "")
        ).sum()
    total_scenarios = max(total_scenarios, 1)
    current_scenario = 0
    
    for i in range(1, 6):  # For each scenario column (1-5)
        answers = []
        
        column_name = f'Scenario No. {i}'
        for idx, row in raw_df.iterrows():
            scenario = row.get(column_name, "")
            
            if pd.isna(scenario) or str(scenario).strip() == "":
                answers.append("")
                continue
            
            # Update progress
            current_scenario += 1
            if progress_callback:
                progress_callback(
                    current_scenario, 
                    total_scenarios, 
                    f"Getting answer for Topic {idx+1}, Scenario {i}"
                )
            
            try:
                # Get model's answer
                answer = client.get_model_answer(scenario, model_name)
                answers.append(answer)
                
                # Delay between requests
                time.sleep(config.DELAY_BETWEEN_REQUESTS)
                
            except Exception as e:
                print(f"Error getting answer for row {idx}, scenario {i}: {str(e)}")
                answers.append(f"ERROR: {str(e)}")
        
        # Add answers column
        result_df[f'Answer_{i}'] = answers
    
    return result_df