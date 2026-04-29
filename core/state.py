from typing import List, Sequence, Annotated, TypedDict, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import io
import base64
import re
import matplotlib.pyplot as plt
import pandas as pd
from duckduckgo_search import DDGS

class NaviState(TypedDict):
    # --- Core Communication ---
    user_input: str             
    task: str                   
    final_answer: str           
    history: Annotated[Sequence[BaseMessage], add_messages]
    
    # --- Persistence & Memory Layer (New) ---
    memory_context: str          # Facts retrieved from Pinecone to guide the current task
    memories_to_save: List[str]  # Key insights from current session to be vectorized later
    
    # --- Technical Data & Skills ---
    plan: List[str]
    generated_tool_code: str
    packages: List[str]          
    inventory: List[str]         
    last_error: str
    is_continue: bool
    aggregated_research: str
    research_notes: str
    image_payload: List[str]     
    current_skill_id: str
    
    # --- Metacognitive & Routing Flags ---
    retry_count: int
    loop_count: int              
    consecutive_failures: int    
    consecutive_research_failures: int 
    meditation_triggered: bool   
    meditation_notes: str        
    is_terminal: bool   
    failure_count: int         
    is_conversational: bool      
    execution_logs: str


import io
import base64
import re
import matplotlib.pyplot as plt
import pandas as pd
from ddgs import DDGS

class NaviEngine:
    @staticmethod
    def run_search(query, max_results=5):
        """Hardcoded DDGS search logic with index safety."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [r['href'] for r in results if 'href' in r]
        except Exception as e:
            print(f"Search Error: {e}")
            return []

    @staticmethod
    def robust_extract(text, keyword):
        """Standardized regex to catch numbers, currency, and scales (B/M/K)."""
        if not text or not isinstance(text, str): return 0.0
        # Look for the keyword, then optional punctuation/spaces, then the number
        pattern = rf"{keyword}[:\-\s]*\$?([\d,.]+)\s*([BMK])?"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                num_str = match.group(1).replace(',', '')
                num = float(num_str)
                suffix = match.group(2)
                if suffix:
                    suffix = suffix.upper()
                    if suffix == 'B': num *= 1e9
                    elif suffix == 'M': num *= 1e6
                    elif suffix == 'K': num *= 1e3
                return num
            except:
                return 0.0
        return 0.0

    @staticmethod
    def safe_calculate(formula):
        """Evaluates simple math strings while blocking dangerous Python commands."""
        # Only allow numbers, math operators, and dots
        clean_formula = re.sub(r'[^0-9./*+\-() ]', '', formula)
        try:
            # Using eval on a sanitized string is safe for simple math
            return eval(clean_formula)
        except Exception as e:
            print(f"Calculation Error: {e}")
            return 0.0

    @staticmethod
    def generate_table(data_context):
        """Converts the internal storage dict into a clean Markdown table."""
        try:
            # Filter out internal keys like 'calculations' or 'formatted_table'
            core_data = {k: v for k, v in data_context.items() 
                         if isinstance(v, dict) and k not in ['calculations']}
            if not core_data:
                return "No data available for table."
            
            df = pd.DataFrame(core_data).T
            return df.to_markdown()
        except Exception as e:
            return f"Table Generation Error: {e}"

    @staticmethod
    def generate_viz(data_context, config):
        """Universal Visualizer that can pull from main storage or calculations."""
        plt.figure(figsize=(10, 6))
        
        metric = config.get('metric')
        labels = []
        values = []

        # Logic to find the metric in either main storage or the calculation sub-dict
        for label, content in data_context.items():
            if label == 'calculations': continue
            if metric in content:
                labels.append(label)
                values.append(content[metric])
            elif 'calculations' in data_context and metric in data_context['calculations']:
                # This allows plotting of derived metrics like 'PE_Ratio'
                labels.append(label)
                values.append(data_context['calculations'][metric])

        if not values:
            plt.close()
            return ""

        if config.get('type') == 'bar':
            plt.bar(labels, values, color='#3498db')
        elif config.get('type') == 'line':
            plt.plot(labels, values, marker='o', color='#e67e22')
        
        plt.title(config.get('title', f"Comparison: {metric}"))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
