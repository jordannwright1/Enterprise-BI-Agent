from typing import Annotated, List, TypedDict, Optional
import operator
from langchain_core.messages import BaseMessage

class NaviState(TypedDict):
    # Core Communication
    user_input: str             # The raw prompt from the user
    task: str            # The distilled task the planner is working on
    final_answer: str            # The final output (Text or Markdown)
    
    # Technical Data
    plan: List[str]
    generated_tool_code: str
    last_error: str
    research_notes: str
    image_payload: List[str]     # Standardized B64 strings for the UI
    
    # Metacognitive & Routing Flags
    retry_count: int
    consecutive_failures: int    # Tracking code execution fails
    consecutive_research_failures: int # Tracking research dead-ends
    meditation_triggered: bool   # Sentinel to ensure meditation only happens once
    meditation_notes: str        # Storage for the 70B's RCA insights
    is_terminal: bool            # Flag to stop the graph if human help is needed
    is_conversational: bool      # Flag to bypass the tool-use UI
    execution_logs: str
