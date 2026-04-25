from typing import List, Sequence, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class NaviState(TypedDict):
    # --- Core Communication ---
    user_input: str             # The raw prompt from the user
    task: str                   # The distilled task the planner is working on
    final_answer: str           # The final output (Text or Markdown)
    history: Annotated[Sequence[BaseMessage], add_messages]
    
    # --- Technical Data & Skills ---
    plan: List[str]
    generated_tool_code: str
    packages: List[str]          # NEW: Dynamic pip dependencies (e.g., ['duckduckgo-search'])
    inventory: List[str]         # NEW: List of available tools (e.g., ['universal_scraper', 'ddgs_search'])
    last_error: str
    aggregated_research: str
    research_notes: str
    image_payload: List[str]     # Standardized B64 strings for the UI
    current_skill_id: str
    
    # --- Metacognitive & Routing Flags ---
    retry_count: int
    loop_count: int              # NEW: Track how many "Search -> Scrape" cycles have occurred
    consecutive_failures: int    # Tracking code execution fails
    consecutive_research_failures: int 
    meditation_triggered: bool   
    meditation_notes: str        
    is_terminal: bool            # Flag to stop the graph if human help is needed
    is_conversational: bool      # Flag to bypass the tool-use UI
    execution_logs: str
