from typing import List, Sequence, Annotated, TypedDict, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

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
    is_conversational: bool      
    execution_logs: str
