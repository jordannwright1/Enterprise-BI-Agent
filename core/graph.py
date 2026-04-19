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
import groq
from importlib import metadata
import sqlite3
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
    """Installs missing packages safely and reports failures."""
    failed_packages = []
    
    for package in package_list:
        try:
            # Check if package is already available
            __import__(package.replace('-', '_'))
        except ImportError:
            try:
                print(f"📦 System: Installing missing dependency: {package}...")
                # We use check_call, but wrap it to catch the "exit status 1"
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            except subprocess.CalledProcessError as e:
                print(f"❌ System: Failed to install {package}")
                failed_packages.append(package)
            except Exception as e:
                print(f"⚠️ System: Unexpected error installing {package}: {e}")
                failed_packages.append(package)
                
    return failed_packages # Return the list of failures

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
        print("✨ Planner: Task complete. Generating final summary...")
        
        format_prompt = f"""
        You are Navi, a professional AI assistant. 
        User Task: {task}
        Raw Data Found: {final_ans_raw}
        
        Provide a friendly, professional summary of the raw data. Make sure the answer reflects what the user was asking for in the task.

        If the user asks about images, you don't have to explain that you are a text based model and cannot generate images.  The images will be generated but you are not responsible for them. 
        
        List data clearly (tables, reports, or bullets).
        
        """
        
        summary_response = llm_fast.invoke(format_prompt).content.strip()
        
        # Return the summary as the final answer and signal the EXIT
        return {
            "final_answer": summary_response,
            "last_error": None,
            "plan": plan + ["### 🏁 ACTION: EXIT"]
        }

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
    fail_count = state.get("consecutive_failures", 0) + 1
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


    if fail_count >= 3:
        print("🛑 CRITICAL: Maximum research attempts reached. Terminating.")
        return {
            "final_answer": "I apologize, but I've encountered a persistent technical issue after several attempts to self-correct. I am stopping here to prevent an infinite loop.",
            "last_error": None,      
            "generated_tool_code": None, 
            "consecutive_failures": 0,    # Reset count
            "plan": state.get("plan", []) + ["### 🏁 ACTION: EXIT", "### 🛑 TERMINATED"]
        }
    
    print(f"\n🔍 RESEARCHER ACTIVATED (Attempt {fail_count}/2)")

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
        response = llm_fast.invoke(reasoning_prompt).content.strip()
        new_strategy = extract_section(response, "### NEW_STRATEGY_NAME")
        
        return {
        "plan": state.get("plan", []) + [f"### RESEARCH NOTES\n{response}"],
        "consecutive_failures": fail_count, 
        "last_error": None,
        "past_strategies": state.get("past_strategies", []) + [new_strategy],
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

        ### TASK CONTEXT
        {task}

        ### PREVIOUS ATTEMPT
        {previous_code}

        ### RESEARCHER'S RECOMMENDATION

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

        6. **Data Aggregation:** If your code generates large datasets (like 1,000 simulation rows), DO NOT return the raw list. 
        Calculate the summary statistics (mean, median, min, max, etc.) INSIDE the 'execute_tool' function. 
        Only return a concise string summary and the Base64 plot. NEVER print massive loops to the console.

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
    ### ROLE: Senior Python Engineer (Specialist in Data Science & Automation)
    
    ### TASK
    {task}

    ### PREVIOUS ATTEMPT & ERROR
    {f"CODE: {previous_code}\nERROR: {last_error}" if previous_code else "Fresh Attempt."}

    ### RESEARCHER'S GUIDANCE
    {research_notes if research_notes else "Use standard best practices for this domain."}

    ### MANDATORY EXECUTION RULES (CRITICAL):
    1. **NO VERBOSE PRINTING:** If the task involves simulations (Monte Carlo), loops, or large datasets, NEVER print individual iteration results. 
    2. **INTERNAL AGGREGATION:** Perform all math/loops inside `execute_tool`. Calculate the final Mean, Median, Std Dev, and Percentiles locally.
    3. **OUTPUT LIMIT:** The string returned by `execute_tool()` MUST be a concise summary (max 500 words). If you provide a table, show only the first/last 5 rows if the data is large.
    4. **VISUALIZATION:** If plotting, use `plt.switch_backend('Agg')`. Return the Base64 string at the VERY end of the result string.
    5. **BS4/SCRAPING:** If scraping, use defensive `if element else "N/A"` checks to avoid NoneType errors. Use Absolute URLs.
    6. **Multi-Plot Handling:** If the user asks for multiple charts, use `plt.figure()` before each plot to ensure they are captured as distinct images. Return all generated Base64 strings sequentially.

    ### STYLE REQUIREMENTS:
    - Use `plt.style.use('ggplot')` for all charts.
    - If calculating stock/finance data, use `pandas` and `numpy`.
    - Format large numbers with commas (e.g., 10,000) for readability.

    ### FINAL STRUCTURE:
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import io, base64

    def execute_tool():
        # 1. Run simulation/logic
        # 2. Aggregate data into a summary string
        # 3. Create plot if needed
        # 4. Return ONLY the summary and the image string
        return summary_string
    ```
    """
    try:
        response = llm_pro.invoke(prompt)
        code = extract_clean_code(response.content)
    except groq.RateLimitError:
            print("⚠️ 70B Rate Limit! Falling back to 8B...")
            response = llm_fast.invoke(prompt)
            code = extract_clean_code(response.content)
    except Exception as e:
    # Generic fallback for other types of "out of tokens" or context errors
        if "rate_limit" in str(e).lower() or "context_length" in str(e).lower():
            print("⚠️ Token/Context limit hit. Falling back to 8B...")
            response = llm_fast.invoke(prompt)
            code = extract_clean_code(response.content)
        else:
            raise e

    if not code:
        return {"last_error": "### ❌ Error\nNo Python code found."}

    try:
        ast.parse(code)
        
        # --- DYNAMIC DEPENDENCY & SAVING ---
        final_packages = extract_dependencies(code)

        if final_packages:
            # Capture the list of packages that failed to install
            failed_installs = ensure_packages(final_packages)
            
            # If any package failed, we halt and report to the Planner
            if failed_installs:
                error_msg = f"### ❌ Dependency Error\nCould not install: {', '.join(failed_installs)}. Please research an alternative library or approach."
                print(f"⚠️ {error_msg}")
                return {
                    "last_error": error_msg,
                    "plan": state.get('plan', []) + [f"### ⚠️ Install Failed: {failed_installs}"]
                }

        # COMMIT TO DATABASE (Only if dependencies passed)
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
    except Exception as e:
        # Catch-all safety net for the creator node
        return {"last_error": f"### ❌ Skill Creator Error\n{str(e)}"}
        


# --- Node 3: Executor ---
import subprocess
import tempfile
import textwrap
import sys
import os
import re

def executor_node(state: NaviState):
    print("\n🚀 [SUBPROCESS] Starting Lite Execution...")
    
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

    # 2. DYNAMIC PACKAGE INSTALLATION (Streamlit Cloud Compatible)
    if packages:
        print(f"📦 Checking/Installing dependencies: {', '.join(packages)}")
        for pkg in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            except Exception as e:
                print(f"⚠️ Warning: Could not install {pkg}: {e}")

    # 3. EXECUTION VIA SUBPROCESS
    # We use a temporary file to hold the script for the sub-process to run
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
        f.write(full_script)
        temp_path = f.name

    print(f"\n{'='*20} EXECUTION START {'='*20}")
    try:
        # Run the script using the same Python interpreter as the main app
        process = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        
        # Combine stdout and stderr to capture all logs and errors
        raw_logs = process.stdout + "\n" + process.stderr
        
        # --- THE LOG SCRUBBER ---
        output_cleaned = re.sub(r"http[s]?://\S+usercontent\.\S+", "", raw_logs)
        output_cleaned = re.sub(r"\[notice\].*?|WARNING: Running pip.*?|immersive_entry_chip", "", output_cleaned)
        output_cleaned = output_cleaned.strip()

        print(output_cleaned)
        print(f"{'='*21} EXECUTION END {'='*21}\n")

        # 4. PARSE MARKERS
        match = re.search(r"---NAVI_RESULT_START---\s*(.*?)\s*---NAVI_RESULT_END---", output_cleaned, re.DOTALL)
        
        if match:
            extracted_data = match.group(1).strip()
            
            # 1. MULTI-STRIPE HARVESTER
            # Captured images are stored in a list to support multiple visuals per task
            image_payloads = []
            # This pattern captures Base64 even with internal newlines/whitespace
            b64_pattern = r"(iVBORw0KGgoAAAANSUhEUg[A-Za-z0-9\+/\s\n=]+)"
            
            # Find all occurrences of Base64 images
            all_figs = re.findall(b64_pattern, extracted_data)
            
            clean_for_llm = extracted_data
            for idx, fig_raw in enumerate(all_figs):
                # Clean the specific match for the UI (remove whitespace/newlines)
                image_data_clean = re.sub(r"\s+", "", fig_raw)
                image_payloads.append(image_data_clean)
                
                # Identify and replace the 'container' (HTML tags, labels like Figure:)
                # We use re.escape to ensure the raw Base64 doesn't break the regex engine
                container_pattern = r"(<img[^>]*?|Figure:\s*|Plot:\s*)?" + re.escape(fig_raw) + r"([^>]*?>)?"
                clean_for_llm = re.sub(container_pattern, f"\n\n[IMAGE_DATA_HIDDEN_{idx}]\n\n", clean_for_llm)

            # 2. FINAL TEXT SAFETY NET (Truncate the "Wall of Numbers")
            # If the Monte Carlo simulation printed 1,000 lines of text, we snip them here.
            if len(clean_for_llm) > 5000:
                header = clean_for_llm[:2500]
                footer = clean_for_llm[-2500:]
                clean_for_llm = f"{header}\n\n... [HEAVY DATA TRUNCATED FOR CONTEXT LIMITS] ...\n\n{footer}"

            # --- VALIDATION: SHORT/EMPTY DATA ---
            if len(clean_for_llm) < 30 and "error" not in clean_for_llm.lower():
                return {
                    "last_error": "### ⚠️ Logic Failure\nExtracted data was too short or empty.",
                    "final_answer": None,
                    "retry_count": retry_count,
                    "plan": current_plan + ["### ⚠️ Truncated Result"]
                }

            # --- VALIDATION: SOFT ERRORS ---
            soft_error_keywords = [
                "failed to", "error:", "none type", "empty", 
                "not found", "division by zero", "an error occurred:", 
                "syntax error", "invalid syntax", "exception", "error sending email:"
            ]
            if any(k in clean_for_llm.lower() for k in soft_error_keywords):
                print("⚠️ Logic Failure detected in extracted data. Re-routing to Planner.")
                return {
                    "last_error": f"### ⚠️ Logic Failure\n{clean_for_llm}",
                    "final_answer": None,
                    "retry_count": retry_count,
                    "plan": current_plan + ["### ⚠️ Execution Logic Failed"]
                }
            
            # --- CLEAR SUCCESS ---
            has_images = len(image_payloads) > 0
            print(f"✅ EXECUTOR SUCCESS: Data extracted ({len(clean_for_llm)} text chars, Images Captured: {len(image_payloads)})")
            return {
                "final_answer": clean_for_llm,
                "image_payload": image_payloads,
                "last_error": None,
                "retry_count": retry_count,
                "consecutive_failures": 0,
                "generated_tool_code": state.get("generated_tool_code"),
                "plan": state.get("plan", []) + ["### ✅ Execution Success"]
            }

        # 5. FAILURE PATH (No Markers = Crash)
        log_lines = output_cleaned.splitlines()
        final_crash_log = "\n".join(log_lines[-15:]).strip() 

        print(f"❌ EXECUTOR CRASH: Returning logs to Planner.")
        return {
            "last_error": f"### ❌ Execution Crash\n```text\n{final_crash_log}\n```",
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
        # Cleanup the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


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
    natural_response = llm_fast.invoke(format_prompt).content.strip()
    return {
            "final_answer": natural_response,
            "plan": ["### ✅ COMPLETE"]
        }

# --- Conditional Routing Logic ---
def route_after_plan(state: NaviState):
    plan = state.get("plan", [])
    if not plan: return "skill_creation"
    
    last_step = str(plan[-1]).upper()
    print(f"DEBUG [Router] - Deciding path for: {last_step}")

    if "EXIT" in last_step or "TERMINATED" in last_step:
        return END
    
    if "RESEARCH" in last_step:
      print("🎯 Router: Match! Redirecting to Research Node.") 
      return "research"
    
    return "skill_creation"


def route_after_execution(state: NaviState):
    last_err = state.get("last_error")
    raw_ans = str(state.get("final_answer", "")).lower()
    
    # If the node flagged an error OR the string looks like an error
    if last_err or "error occurred" in raw_ans or "occurred" in raw_ans:
        print(f"🎯 Router: Redirecting to Planner for error handling.")
        return "planner"
    
        
    return "planner"


# --- Node Configuration ---
workflow = StateGraph(NaviState)

workflow.add_node("planner", planner_node)
workflow.add_node("skill_creation", skill_creator_node)
workflow.add_node("executor", executor_node)
workflow.add_node("research", research_node)

workflow.set_entry_point("planner")

# --- Planner Routing ---
workflow.add_conditional_edges(
    "planner", 
    route_after_plan, 
    {
        "skill_creation": "skill_creation", 
        "research": "research", 
        END: END 
    }
)

# --- Technical Edges ---
workflow.add_edge("research", "skill_creation")
workflow.add_edge("skill_creation", "executor")

# --- Executor Routing ---
workflow.add_conditional_edges(
    "executor", 
    route_after_execution, 
    {
        "planner": "planner" # Success or Failure, go to Planner to decide next move
    }
)

# Compile
navi_app = workflow.compile(checkpointer=MemorySaver())
