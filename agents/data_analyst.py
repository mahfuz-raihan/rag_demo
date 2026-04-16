import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.llm import get_llm
from core.session_store import session_store

def data_analyst_node(state):
    """Generates Python code to create Plotly charts based on user queries."""
    print("--- [Node: Data Analyst] Generating Chart ---")
    llm = get_llm()
    
    question = state.get("question")
    session_id = state.get("session_id")
    
    # 1. Retrieve the dataframe from our session store
    df = session_store.get_dataframe(session_id)
    if df is None:
        return {"generation": "I couldn't find any data. Please upload a CSV or Excel file first.", "figures": []}
    
    # 2. Instruct the LLM to write Plotly code
    prompt = f"""
    You are an expert Python Data Analyst. 
    You have a pandas dataframe loaded as a variable named `df`. 
    
    Data Schema and Types:
    {df.dtypes.to_string()}
    
    First 3 rows of data:
    {df.head(3).to_string()}
    
    The user asked: "{question}"
    
    Task: Write Python code using `plotly.express` (which is already imported as `px`) to create an interactive chart that answers the user's question.
    
    CRITICAL RULES:
    1. You must assign the resulting chart to a variable exactly named `fig`.
    2. Do NOT use `fig.show()`.
    3. Output ONLY valid Python code inside a ```python ``` codeblock. Do not add any conversational text.
    """
    
    response = llm.invoke(prompt)
    code_text = response.content
    
    # 3. Extract the python code from the markdown block
    match = re.search(r"```python\n(.*?)\n```", code_text, re.DOTALL)
    code = match.group(1) if match else code_text.replace("```python", "").replace("```", "")
    
    # 4. Safely execute the code in an isolated environment
    local_vars = {"df": df, "px": px, "go": go}
    try:
        # Note: exec() executes the string as Python code. 
        # local_vars acts as the environment, so 'fig' will be saved inside it.
        exec(code, {}, local_vars)
        fig = local_vars.get("fig")
        
        if fig:
            return {
                "generation": "Here is the interactive chart based on your data:",
                "figures": [fig]
            }
        else:
            return {"generation": "I analyzed the data, but failed to generate the visual chart.", "figures": []}
            
    except Exception as e:
        return {"generation": f"I tried to write code to chart this, but encountered an error: {str(e)}", "figures": []}