import pandas as pd
import os
from config import config

def load_scenarios(file_path=None):
    """
    Load scenarios from Excel file dynamically.
    
    Args:
        file_path: Optional path to Excel file. If None, uses config.SCENARIOS_FILE
        
    Returns:
        DataFrame with columns: Criterion ID, Criterion, Scenario_1, ..., Scenario_5
    """
    if file_path is None:
        # Try to find the scenarios file
        possible_files = [
            os.path.join(config.DATA_DIR, 'Criterinos and 100 Scenarios.xlsx'),
            os.path.join(config.DATA_DIR, 'Criterions and 100 Scenarios.xlsx'),
            config.SCENARIOS_FILE,
        ]
        
        for file in possible_files:
            if os.path.exists(file):
                file_path = file
                break
        
        if file_path is None:
            raise FileNotFoundError(f"Could not find scenarios file. Tried: {possible_files}")
    
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Normalize column names (handle variations)
    column_mapping = {}
    scenario_cols_found = []
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        col_str = str(col).strip()
        
        # Map Criterion ID
        if 'criterion id' in col_lower or 'criterion_id' in col_lower or col_str.startswith('Criterion ID'):
            column_mapping[col] = 'Criterion ID'
        # Map Criterion (but not Criterion ID)
        elif 'criterion' in col_lower and 'id' not in col_lower:
            if col not in column_mapping:  # Don't overwrite if already mapped
                column_mapping[col] = 'Criterion'
        # Map Scenario columns
        elif 'scenario' in col_lower:
            scenario_cols_found.append((col, col_lower))
    
    # Process scenario columns more carefully
    scenario_mapping = {}  # Maps number -> original column name
    used_cols = set()  # Track which columns have been mapped
    
    for col, col_lower in scenario_cols_found:
        if col in used_cols:
            continue
        
        col_str = str(col).strip()
        matched = False
        
        # Try to find scenario number
        for i in range(1, 6):
            # Check various patterns
            patterns = [
                f'scenario no. {i}',
                f'scenario no {i}',
                f'scenario {i}',
                f'scenario_{i}',
                f'scenario{i}',
                f's{i}',
            ]
            # Also check if number appears after "scenario" keyword
            if any(pattern in col_lower for pattern in patterns) or (f' {i}' in col_str and 'scenario' in col_lower):
                if i not in scenario_mapping:  # Use first match for each number
                    scenario_mapping[i] = col
                    column_mapping[col] = f'Scenario_{i}'
                    used_cols.add(col)
                    matched = True
                    break
        
        # If no exact match, try to match by position in the list
        if not matched and len(scenario_cols_found) <= 5:
            idx = next((i for i, (c, _) in enumerate(scenario_cols_found) if c == col), -1)
            if idx >= 0 and (idx + 1) not in scenario_mapping:
                scenario_mapping[idx + 1] = col
                column_mapping[col] = f'Scenario_{idx + 1}'
                used_cols.add(col)
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns, create empty ones if missing
    if 'Criterion ID' not in df.columns:
        # Try to find alternative
        for col in df.columns:
            if 'id' in str(col).lower():
                df['Criterion ID'] = df[col]
                break
        else:
            df['Criterion ID'] = range(1, len(df) + 1)
    
    if 'Criterion' not in df.columns:
        # Try to find alternative
        for col in df.columns:
            if 'criterion' in str(col).lower() or 'topic' in str(col).lower():
                df['Criterion'] = df[col]
                break
        else:
            df['Criterion'] = ''
    
    # Ensure all scenario columns exist
    for i in range(1, 6):
        if f'Scenario_{i}' not in df.columns:
            df[f'Scenario_{i}'] = ''
    
    # Select and reorder columns
    result_cols = ['Criterion ID', 'Criterion'] + [f'Scenario_{i}' for i in range(1, 6)]
    df = df[[col for col in result_cols if col in df.columns]]
    
    return df


def get_scenarios_by_category(scenarios_df=None):
    """
    Organize scenarios by Sub-Focus Area (category).
    
    Args:
        scenarios_df: Optional DataFrame. If None, loads from file.
        
    Returns:
        Dictionary: {category_code: {'name': str, 'criterions': list of dicts}}
    """
    if scenarios_df is None:
        scenarios_df = load_scenarios()
    
    categories_data = {}
    
    for category, (start_idx, end_idx) in zip(config.CATEGORIES, config.CATEGORY_RANGES):
        # Get criteria for this category (0-indexed, so start_idx is inclusive, end_idx is exclusive)
        category_rows = scenarios_df.iloc[start_idx:end_idx].copy()
        
        criterions = []
        for _, row in category_rows.iterrows():
            criterion_data = {
                'criterion_id': row['Criterion ID'],
                'criterion': row['Criterion'],
                'scenarios': []
            }
            
            # Get all scenarios for this criterion
            for i in range(1, 6):
                scenario_col = f'Scenario_{i}'
                if scenario_col in row and pd.notna(row[scenario_col]) and str(row[scenario_col]).strip():
                    criterion_data['scenarios'].append({
                        'number': i,
                        'text': str(row[scenario_col]).strip()
                    })
            
            criterions.append(criterion_data)
        
        categories_data[category] = {
            'name': config.CATEGORY_NAMES[category],
            'criterions': criterions
        }
    
    return categories_data

