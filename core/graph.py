import os
import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from core.state import NaviState
from langgraph.checkpoint.memory import MemorySaver
import json
import subprocess
import sys
from importlib import metadata
import docker
import sqlite3
from tools.base_search import search
import ast
from core.database import get_skill, save_skill, init_db

def install_package(package):
    try:
        # Modern way to check if a package is installed
        metadata.version(package)
    except metadata.PackageNotFoundError:
        # Only install if missing
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def get_skill_name(task_description: str) -> str:
    """Uses the fast LLM to generate a creative, snake_case codename."""
    prompt = f"Generate a 2-3 word creative snake_case codename for this task: {task_description}. Output ONLY the name."
    response = llm_fast.invoke(prompt)
    # Clean up the output to ensure it's a valid filename/key
    name = re.sub(r'[^a-z0-9_]', '', response.content.lower().replace(" ", "_"))
    return name


def extract_clean_code(content: str) -> str:
    # 1. Primary Extraction
    code_match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
    
    if not code_match:
        fallback_match = re.search(r"((?:import|def|from)\s+.*)", content, re.DOTALL)
        if not fallback_match: return ""
        raw_code = fallback_match.group(1)
    else:
        raw_code = code_match.group(1)

    # 2. Sanitization (Indentation-Safe)
    lines = raw_code.split('\n')
    clean_lines = []
    
    for line in lines:
        # DO NOT strip() the whole line here. Only check the content.
        content_check = line.strip()
        
        if not content_check:
            clean_lines.append("")
            continue
            
        # Fix the specific 'requestsfrom' hallucination
        if "import requestsfrom" in content_check:
            line = line.replace("import requestsfrom", "import requests\nfrom")
            
        # Filter: If it's a "Step 1" or plain English instruction at the start of a line
        # But allow lines that start with spaces (indented code)
        if line and not line.startswith((' ', '\t')):
            python_keywords = ('import', 'from', 'def', 'class', 'if', 'else', 'elif', 
                               'try', 'except', 'with', 'return', 'print', 'raise', '#')
            if not any(content_check.startswith(k) for k in python_keywords) and '=' not in content_check:
                continue
            
        clean_lines.append(line) # Add the original line with its spaces!

    return "\n".join(clean_lines).strip()

import subprocess
import sys

def ensure_packages(package_list):
    """Installs missing packages via pip."""
    for package in package_list:
        try:
            # Check if package is already available
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"📦 System: Installing missing dependency: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def extract_section(text, section_header):
    """
    Extracts content starting from a specific header until the next '###' header.
    """
    # Pattern looks for the header and captures everything until the next '###' or end of string
    pattern = rf"{re.escape(section_header)}\s*(.*?)(?=\n###|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def delete_skill(keyword):
    conn = sqlite3.connect("tools/navi_skills.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM skills WHERE keyword = ?", (keyword,))
    conn.commit()
    conn.close()


# --- LLM Setup ---
# 70B for Planning/Learning, 8B for quick Execution checks
llm_pro = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_fast = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- Node 1: Planner ---
def planner_node(state: NaviState):
    final_ans_raw = state.get("final_answer")
    last_error = state.get("last_error")
    retry_count = state.get("retry_count", 0)
    plan = state.get("plan", [])
    task = state.get("task")

    # helper for scannability
    last_step = str(plan[-1]).upper() if plan else ""

    if final_ans_raw and not last_error:
        return {"plan": plan + ["### 📢 ACTION: SUMMARIZE"]}

    if last_error:
        if retry_count < 1:
            print(f"🔄 Planner: Attempt {retry_count + 1} failed. Retrying CODE.")
            return {
                "plan": plan + ["### 🛠️ ACTION: CODE"],
                "retry_count": retry_count + 1
            }
        else:
            print("🚨 Planner: Informed attempts failed. Returning to RESEARCH for new strategy.")
            return {
                "plan": plan + ["### 🔍 ACTION: RESEARCH"],
                "retry_count": 0 
            }

    # POST-RESEARCH ROUTING
    if "RESEARCH" in last_step:
        print("💡 Planner: Research complete. Routing to Skill Creator.")
        return {
            "plan": plan + ["### 🛠️ ACTION: CODE"],
            "retry_count": 0,
            "generated_tool_code": state.get("generated_tool_code") 
        }
    
    # INITIAL START - SKILL CHECK
    # We check for a pre-existing skill before defaulting to a fresh generation
    skill_id = get_skill_name(task) # Assuming you have this helper
    existing_skill = get_skill(skill_id) 
    
    if existing_skill and not plan: # Only load on the very first step
        print(f"🧠 ### 💾 ACTION: LOAD_SKILL")
        print(f"✅ Planner: Skill '{skill_id}' found in library. Loading successfully.")
        return {
            "plan": plan + ["### 💾 ACTION: LOAD_SKILL"],
            "generated_tool_code": existing_skill['code'],
            "packages": existing_skill.get('packages', []),
            "retry_count": 0
        }

    # DEFAULT (Initial Start)
    return {"plan": plan + ["### 🛠️ ACTION: CODE"], "retry_count": 0}

def extract_dependencies(code):
    # 1. Extract all top-level imports and 'from x import y'
    # Pattern looks for 'import pkg' or 'from pkg ...'
    raw_imports = re.findall(r"^(?:import|from)\s+([a-zA-Z0-9_]+)", code, re.MULTILINE)
    
    # 2. Define a map for packages where the import name != pip install name
    package_map = {
        "bs4": "beautifulsoup4",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml"
    }
    
    # 3. Filter out Python Standard Library modules (so we don't try to 'pip install os')
    # Use sys.stdlib_module_names (Python 3.10+)
    std_libs = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else []
    
    final_packages = []
    for imp in set(raw_imports):
        # Map to pip name, or keep original
        pkg = package_map.get(imp, imp)
        # Only include if it's not a built-in standard library
        if pkg not in std_libs and pkg != 'execute_tool':
            final_packages.append(pkg)
            
    return final_packages

def research_node(state: NaviState):
    task = state.get("task", "Unknown Task")
    raw_error = state.get('last_error')
    final_answer = state.get('final_answer')
    past_strategies = state.get("past_strategies", [])
    failed_code = state.get("generated_tool_code")
    # 1. Error Identification
    if raw_error is None:
        if final_answer:
            last_error = f"Logic Failure: Data was returned but deemed incomplete: {str(final_answer)[:200]}"
        else:
            last_error = "Unknown error: No output captured."
    else:
        last_error = str(raw_error)
    
    print(f"\n🔍 RESEARCHER ACTIVATED: Analyzing failure...")

    # 2. Strategic Pivot Prompt
    reasoning_prompt = f"""
    ### 🧠 META-ANALYSIS TASK
    Objective: {task}
    
    ### 🚫 FAILED ATTEMPTS
    Architectural patterns tried: {past_strategies if past_strategies else "None"}
    Failed Code:  {failed_code if failed_code else "No code generated yet."}
    Last error encountered: {last_error}

    ### 🛠️ STRATEGY PIVOT
    1. ANALYZE: Why did the previous logic fail?
    2. DIVERGE: Propose a fundamentally different technical approach.
       - If CSS failed, try Regex or text-based finding.
       - If specific elements were None, propose a more defensive container search.
    
    ### 📋 OUTPUT FORMAT
    ### NEW_STRATEGY_NAME
    <Approach Title>
    ### DIAGNOSIS
    <Why the old code failed>
    ### RECOMMENDED_CODE
    <Pure Python implementation using requests/BS4>
    """

    try:
        response = llm_pro.invoke(reasoning_prompt).content.strip()
        new_strategy = extract_section(response, "### NEW_STRATEGY_NAME")
        
        return {
            "plan": state.get("plan", []) + [f"### RESEARCH NOTES\n{response}"],
            "past_strategies": past_strategies + [new_strategy if new_strategy else "Alternative Attempt"],
            "last_error": None, # Reset to allow the Planner to move to Code
            "retry_count": state.get("retry_count", 0)
        }
    except Exception as e:
        return {"plan": state.get("plan", []) + [f"### RESEARCH FAILED: {str(e)}"]}
        

# --- Node 2: Skill Creator (The Self-Learning Node) ---

import ast
import re
from langchain_core.messages import AIMessage

def skill_creator_node(state: NaviState):
    task = state.get('task', "")
    plan = state.get('plan', [])
    last_error = state.get('last_error', "None")
    retry_count = state.get("retry_count", 0)
    research_notes = next((p for p in reversed(plan) if "### RESEARCH NOTES" in p or "### DIAGNOSIS" in p), "")
    previous_code = state.get("generated_tool_code")

    # CONSTRUCT THE PROMPT DYNAMICALLY
    if research_notes:
        # --- PATH A: POST-RESEARCH ATTEMPT ---
        context_instruction = f"""
        ### RESEARCH-LED ATTEMPT ({retry_count}/2)
        The Researcher has identified the cause of the previous 'NoneType' or logic failure.
        
        ### RESEARCH NOTES & RECOMMENDED FIX:
        {research_notes}

        When calculating averages, always check if the list is empty first to avoid ZeroDivisionError. Example: avg = sum(vals)/len(vals) if vals else 0.
        
        ### MANDATORY DESIGN PATTERNS:
        1. **Direct Targeting over Positional Traversal:** NEVER use `.find_next_sibling()` or list indices (e.g., `[0]`) for navigation. 
        ALWAYS search for elements by visible text or partial class matches using Regex.
        *Example:* `link = soup.find('a', string=re.compile(r'CategoryName', re.I))`

        2. **Absolute URL Handling:**
        Always use `urllib.parse.urljoin(base_url, relative_path)` to resolve links found in `href` attributes.

        3. **Defensive Extraction (The "None-Guard"):**
        Every `.find()` call must be checked. Wrap sub-element extraction in a helper function or inline conditional.
        *Example:* `price = item.find('p').text if item.find('p') else "0"`

        4. **Detailed Failure Logs:**
        If a critical element is missing, the code must print a specific diagnostic message before failing.
        *Example:* `if not container: print("DEBUG: Could not find result-container div")`

        5. **No Scrapy in Production:**
        The Researcher may provide Scrapy CSS selectors. You MUST translate these to `requests` + `BeautifulSoup` (using `soup.select` or `soup.find`).

        ### FINAL STRUCTURE:
        ```python
        import requests
        import re
        import pandas as pd
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        def execute_tool():
          base_url = "URL_FROM_TASK"
          # Implement logic...
          return "Final formatted string of results"
        """
    else:
        # --- PATH B: STANDARD FIRST ATTEMPT ---
        context_instruction = "This is a fresh attempt. Create a robust scraping script."
    new_strategy = extract_section(research_notes, "### NEW_STRATEGY_NAME")
    recommended_code = extract_section(research_notes, "### RECOMMENDED_CODE")
    prompt = f"""
    ### ROLE: Senior Python Engineer
    
   ### STRATEGIC SHIFT DETECTED
    The previous approach failed. The Researcher has proposed a new mental model:
    **Strategy:** {new_strategy if new_strategy else "General Diversification"}
    
    ### PREVIOUS ATTEMPT (FAILED)
    {f"CODE:\\n{previous_code}\\n\\nERROR:\\n{last_error}" if previous_code else "No previous attempt."}

    ### RESEARCHER'S INSTRUCTIONS
    {research_notes}

    ### IMPLEMENTATION GUIDELINE
    {recommended_code if recommended_code else "Write a fresh solution from scratch."}
    
    ### TASK
    {task}
    
    ### CRITICAL INSTRUCTIONS
    1. Use the Researcher's logic to find elements.
    2. Wrap the execution in try-except blocks.
    3. Ensure 'execute_tool()' returns a clear summary string of the data found.
    4. Avoid the previous error: {last_error}
    5. TRANSLATE SELECTORS: If the Researcher provided Scrapy-style code (response.css or yield), translate it to BeautifulSoup (soup.select or soup.find) and ensure the function returns a string.


    ### FINAL TOOL STRUCTURE
    ```python
    import requests
    from bs4 import BeautifulSoup
    import re

    def execute_tool():
        # Implementation based on Research
    ```
    """

    response = llm_pro.invoke(prompt)
    code = extract_clean_code(response.content)

    if not code:
        return {"last_error": "### ❌ Error\nNo Python code found."}

    try:
        ast.parse(code)
        
        # --- DYNAMIC DEPENDENCY & SAVING ---
        
        final_packages = extract_dependencies(code)

        if final_packages:
            ensure_packages(final_packages)

        # COMMIT TO DATABASE
        task_id = get_skill_name(task)
        save_skill(task_id, task, code, final_packages)
        print(f"💾 Skill Saved Successfully: {task_id}")

        return {
            "generated_tool_code": code,
            "packages": final_packages,
            "last_error": None,
            "plan": state.get('plan', []) + [f"### 🛠 Skill Saved\nTask: {task_id}"]
        }
    
    except SyntaxError as e:
        return {"last_error": f"### ❌ Syntax Error\n{str(e)}"}
        

        
client = docker.from_env()

# --- Node 3: Executor ---
import textwrap
import base64
import re

def executor_node(state: NaviState):
    print("\n🚀 [DOCKER] Starting Container Execution...")
    
    code = state.get('generated_tool_code')
    packages = state.get('packages', []) 
    retry_count = state.get('retry_count', 0)
    current_plan = state.get('plan', [])

    if not code:
        return {
            "last_error": "### ❌ Execution Failed\nNo code found.",
            "retry_count": retry_count + 1
        }

    # 1. WRAP & DEDENT (Ensuring strict output markers)
    raw_script = f"""
import sys
import io

# Force UTF-8 and unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

{code}

if __name__ == "__main__":
    try:
        result = execute_tool()
        output = str(result) if result is not None else "Error: None"
        
        # Dedicated print calls with flushing for the parser
        sys.stdout.write("\\n---NAVI_RESULT_START---\\n")
        sys.stdout.write(output)
        sys.stdout.write("\\n---NAVI_RESULT_END---\\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"EXECUTION_ERROR: {{e}}")
        sys.stdout.flush()
"""    
    full_script = textwrap.dedent(raw_script).strip()

    # 2. ENCODE & PREPARE
    encoded_payload = base64.b64encode(full_script.encode('utf-8')).decode('utf-8')
    pkg_str = " ".join(packages)
    install_cmd = f"pip install --no-cache-dir {pkg_str} --quiet --root-user-action=ignore &&"
    docker_command = f"{install_cmd} echo {encoded_payload} | base64 -d | python3"

    print(f"\n{'='*20} DOCKER START {'='*20}")
    container = None
    try:
        container = client.containers.run(
            "python:3.11-slim",
            command=["sh", "-c", docker_command],
            detach=True,
            network_mode="bridge",
            dns=["8.8.8.8", "1.1.1.1"],
            mem_limit="512m",
            environment={"PYTHONIOENCODING": "utf-8"}
        )

        container.wait(timeout=60)
        raw_logs = container.logs().decode("utf-8", errors="replace")
        
        # --- THE LOG SCRUBBER ---
        output_cleaned = re.sub(r"http[s]?://\S+usercontent\.\S+", "", raw_logs)
        output_cleaned = re.sub(r"\[notice\].*?|WARNING: Running pip.*?|immersive_entry_chip", "", output_cleaned)
        output_cleaned = output_cleaned.strip()

        print(output_cleaned)
      
        print(f"{'='*21} DOCKER END {'='*21}\n")

        # 3. PARSE MARKERS
        match = re.search(r"---NAVI_RESULT_START---\s*(.*?)\s*---NAVI_RESULT_END---", output_cleaned, re.DOTALL)
        
        if match:
            extracted_data = match.group(1).strip()
            
            # --- VALIDATION: SHORT/EMPTY DATA ---
            if len(extracted_data) < 10 and "error" not in extracted_data.lower():
                return {
                    "last_error": "### ⚠️ Logic Failure\nExtracted data was too short or empty.",
                    "final_answer": None,
                    "retry_count": retry_count, # Keep count to trigger Research
                    "plan": current_plan + ["### ⚠️ Truncated Result"]
                }

            # --- VALIDATION: SOFT ERRORS (Logic/Scraping Failure) ---
            soft_error_keywords = [
            "failed to", "error:", "none type", "empty", 
            "none", "not found", "division by zero"
          ]
            if any(k in extracted_data.lower() for k in soft_error_keywords):
                print("⚠️ Logic Failure detected in extracted data. Re-routing to Planner.")
                return {
                    "last_error": f"### ⚠️ Logic Failure\n{extracted_data}",
                    "final_answer": None,
                    "generated_tool_code": None, # Force re-generation
                    "retry_count": retry_count,
                    "plan": current_plan + ["### ⚠️ Execution Logic Failed"]
                }
            
            # --- CLEAR SUCCESS ---
            print(f"✅ EXECUTOR SUCCESS: Data extracted ({len(extracted_data)} chars)")
            return {
                "final_answer": str(extracted_data),
                "last_error": None,
                "retry_count": retry_count,
                "generated_tool_code": state.get("generated_tool_code"),
                "plan": ["### ✅ Execution Success"]
            }

        # 4. FAILURE PATH (No Markers = Crash)
        log_lines = output_cleaned.splitlines()
        final_crash_log = "\n".join(log_lines[-15:]).strip() 

        print(f"❌ EXECUTOR CRASH: Returning logs to Planner.")
        return {
            "last_error": f"### ❌ Container Crash\n```text\n{final_crash_log}\n```",
            "final_answer": None,
            "generated_tool_code": state.get("generated_tool_code"),
            "retry_count": retry_count,
            "plan": current_plan + [f"### ❌ Attempt {retry_count + 1} Crashed"]
        }

    except Exception as e:
        print(f"🏗️ INFRASTRUCTURE ERROR: {str(e)}")
        return {
            "last_error": f"### 🏗️ Infrastructure Error\n{str(e)}", 
            "final_answer": None,
            "retry_count": retry_count,
            "plan": current_plan + ["### 🏗️ Infra Error"],
            "generated_tool_code": state.get("generated_tool_code")
        }
    
    finally:
        if 'container' in locals() and container:
            try:
                container.remove(force=True)
            except:
                pass
            


# --- Node 4: Human-in-the-Loop ---
def human_gate_node(state: NaviState):
    """Safety buffer for sensitive actions."""
    # This node is a placeholder. LangGraph 'interrupt' handles the actual pause.
    return {"history": [AIMessage(content="Waiting for human authorization...")]}

def summarizer_node(state: NaviState):
    print("✨ Summarizer Node: Transforming raw data into conversation...")
    raw_data = state.get("final_answer")
    task = state.get("task")
    
    format_prompt = f"""
    You are a helpful assistant named Navi. The user asked: {task}. 
    I have the raw result here: {raw_data}.
    
    Transform this into a friendly, professional, and clear conversational response. 
    Summarize any lists or data found. Do not mention 'raw data' or 'markers'.  
    When listing items, use a bullet or numbered list, or a table if applicable.
    Do NOT list results in one sentence, list each result on a new line.

    BAD example: Linear Regression Model: y = 7.89x + 6.26 R-squared: 1.00 Predicted Revenue for 50,000 Marketing Spend: 400.65

    GOOD example: 
    Linear Regression Model: y = 7.89x + 6.26 
    R-squared: 1.00
    Predicted Revenue for 50,000 Marketing Spend: 400.65
    """
    
    try:
        natural_response = llm_fast.invoke(format_prompt).content.strip()
        return {
            "final_answer": natural_response,
            "plan": ["### ✅ COMPLETE"]
        }
    except:
        # Fallback if LLM fails
        return {"plan": ["### ✅ COMPLETE"]}

# --- Conditional Routing Logic ---
def route_after_plan(state: NaviState):
    plan = state.get("plan", [])
    if not plan: return "skill_creation"
    
    last_step = str(plan[-1]).upper()
    print(f"DEBUG [Router] - Deciding path for: {last_step}")

    if "SUMMARIZE" in last_step: return "summarizer"
    if "RESEARCH" in last_step:
      print("🎯 Router: Match! Redirecting to Research Node.") 
      return "research"
    
    return "skill_creation"


def route_after_execution(state: NaviState):
    """
    This ensures that if the Executor returned a last_error, 
    we go back to the Planner, NOT the Summarizer.
    """
    last_err = state.get("last_error")
    
    if last_err:
        print(f"🔄 Router: Error detected ({last_err[:30]}...). Returning to Planner.")
        return "planner"
    return "planner"

# --- Graph Construction ---
workflow = StateGraph(NaviState)

# Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("skill_creation", skill_creator_node)
workflow.add_node("executor", executor_node)
workflow.add_node("research", research_node)
workflow.add_node("summarizer", summarizer_node)

# Entry
workflow.set_entry_point("planner")

# Conditional Routing from Planner
workflow.add_conditional_edges(
    "planner", 
    route_after_plan, 
    {
        "skill_creation": "skill_creation", 
        "research": "research", 
        "summarizer": "summarizer"
        
    }
)

# Technical Loop
workflow.add_edge("skill_creation", "executor")
workflow.add_edge("executor", "planner") # Return to Planner to check results
workflow.add_edge("research", "planner") # Return to Planner to apply research

# Terminal Edge
workflow.add_edge("summarizer", END)

# Compile
navi_app = workflow.compile(checkpointer=MemorySaver())
