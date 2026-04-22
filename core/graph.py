import os
import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from core.state import NaviState
from langgraph.checkpoint.memory import MemorySaver
import json
from langchain_google_genai import ChatGoogleGenerativeAI
import subprocess
import sys
from langchain_ollama import ChatOllama
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
    """Uses the pro LLM to generate a creative, snake_case codename."""
    prompt = f"Generate a 3 word creative snake_case codename for this task: {task_description}. Output ONLY the name.  The name MUST be ONLY 3 words total.  DO NOT include your thoughts or reasoning in the snake case codename you generate."
    try:
        response = llm_pro.invoke(prompt)
    except:
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

    if not text or not isinstance(text, str):
        return "No specific notes provided."
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
llm_gemini_25 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7
)
llm_gemini_20 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7
)
llm_gemini_15 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7
)

llm_qwen_cloud = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.1,  # Keep it ultra-precise for coding
    num_ctx=32768     
)

# --- Node 1: Planner ---
def planner_node(state: NaviState):
    final_ans_raw = state.get("final_answer")
    last_error = state.get("last_error")
    retry_count = state.get("retry_count", 0)
    plan = state.get("plan", [])
    task = state.get("task")

    # helper for scannability
    last_step = str(plan[-1]).upper() if plan else ""
    
    if state.get("meditation_notes"):
        if retry_count >= 2:
            return "Maximum tries reached!  Stopping here to prevent an infinite loop. Please try rephrasing your question"
        return {
            "plan": plan + ["### 🛠️ ACTION: CODE"], "retry_count": state.get("retry_count", 0) + 1
        }
    
    if final_ans_raw and not last_error:
        print("✨ Planner: Task complete. Generating final summary...")
        
        format_prompt = f"""
        You are Navi, a professional AI assistant. 
        User Task: {task}
        Raw Data Found: {final_ans_raw}
        
        Provide a friendly, professional summary of the raw data. Make sure the answer reflects what the user was asking for in the task. Never tell the user you can't produce images if they ask you to provide an image, simply return the data and continue your response.

        Never provide individual data points used to generate graphs.

        Never provide code blocks in your response.

        
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
                "task": task,
                "retry_count": retry_count + 1
            }
        else:
            print("🚨 Planner: Informed attempts failed. Returning to RESEARCH for new strategy.")
            return {
                "plan": plan + ["### 🔍 ACTION: RESEARCH"],
                "task": task,
                "retry_count": 0 
            }

    # POST-RESEARCH ROUTING
    if "RESEARCH" in last_step:
        print("💡 Planner: Research complete. Routing to Skill Creator.")
        return {
            "plan": plan + ["### 🛠️ ACTION: CODE"],
            "retry_count": 0,
            "task": task,
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
            "task": task,
            "packages": existing_skill.get('packages', []),
            "retry_count": 0
        }

    # DEFAULT (Initial Start)
    return {"plan": plan + ["### 🛠️ ACTION: CODE"], "retry_count": 0, "task": task}

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
    res_fails = state.get("consecutive_research_failures", 0) + 1
    logs = state.get("execution_logs")
    research_notes = state.get("research_notes")
    # 1. Error Identification
    if raw_error is None:
        if final_answer:
            last_error = f"Logic Failure: Data was returned but deemed incomplete: {str(final_answer)[:200]}"
        else:
            last_error = "Unknown error: No output captured."
    else:
        last_error = str(raw_error)


    if res_fails >= 2:
        print("🛑 CRITICAL: Maximum research attempts reached.")
        return {
            "consecutive_research_failures": res_fails,
            "plan": state.get("plan", []) + ["### 🏁 ACTION: EXIT"]
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

    ### EXECUTION LOGS
    {logs}

    Focus on the last 10 lines of the execution logs. If it's a ModuleNotFoundError, tell the Skill Creator to add the package to the import block. If it's a ValueError from the scraping logic, tell the Skill Creator the browser structure has shifted.  Always give the fix for the error in your response so that it is passed correctly to the skill creator.

    ### 🛠️ STRATEGY PIVOT
    1. ANALYZE: Why did the previous logic fail?
    2. DIVERGE: Propose a fundamentally different technical approach.
       - If CSS failed, try Regex or text-based finding.
       - If specific elements were None, propose a more defensive container search.
    
    ### 📋 OUTPUT FORMAT
    Respond ONLY in the following JSON format. Ensure the "code" value is a single string with escaped newlines:
    {{
        "diagnosis": "Short paragraph summary of the error",
        "solution_logic": "Two sentence strategy",
        "code": "The corrected python code block"
    }}
    """
    
    try:
        # PRIMARY: Gemini 1.5 Flash
        raw_response = llm_gemini_25.invoke(reasoning_prompt).content.strip()
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}. Falling back to 70B...")
        try:
            # FALLBACK 1: Llama 70B
            raw_response = llm_pro.invoke(reasoning_prompt).content.strip()
        except (groq.RateLimitError, Exception):
            print("⚠️ 70B Limit or Error! Falling back to 8B...")
            # FALLBACK 2: Llama 8B
            raw_response = llm_fast.invoke(reasoning_prompt).content.strip()

    # Clean Markdown Wrappers
    clean_json = re.sub(r'^```json\s*|```$', '', raw_response, flags=re.MULTILINE | re.IGNORECASE).strip()
    
    try:
        # Standard JSON Parse
        data = json.loads(clean_json)
        diagnosis = data.get("diagnosis", "No diagnosis provided.")
        isolated_code = data.get("code", "")
    except Exception as e:
        # Regex Fallback if the JSON structure is malformed
        diag_match = re.search(r'"diagnosis":\s*"(.*?)"', clean_json, re.DOTALL)
        code_match = re.search(r'"code":\s*"(.*?)"', clean_json, re.DOTALL)
        
        diagnosis = diag_match.group(1) if diag_match else f"Parse Error: {str(e)}"
        isolated_code = code_match.group(1) if code_match else clean_json

    
    # 5. Return updated state
    return {
        "research_notes": isolated_code, # Sent to Skill Creator
        "last_error": f"### 🔍 DIAGNOSIS\n{diagnosis}", # Sent to Streamlit UI
        "consecutive_research_failures": state.get("consecutive_research_failures", 0) + 1,
        "plan": state.get("plan", []) + [f"### 🔍 RESEARCH COMPLETE: {diagnosis}"]
    }
        

# --- Node 2: Skill Creator (The Self-Learning Node) ---

import ast
import re
from langchain_core.messages import AIMessage

def skill_creator_node(state: NaviState):
    task = state.get('task', "")
    plan = state.get('plan', [])
    last_error = state.get('last_error', "None")
    retry_count = state.get("retry_count", 0)
    research_notes = state.get("research_notes")
    meditation_notes = state.get("meditation_notes") if state.get("meditation_notes") else ""
    previous_code = state.get("generated_tool_code")
    error_msg = ""
    prompt = f"""
    ### ROLE: Principal AI Automation Architect
    
    ### CONTEXT
    You are the skill creator of Navi, an autonomous agent. You write self-contained, robust Python functions (`execute_tool`) to solve high-stakes technical tasks.

    ### MISSION
    {task}

    ### DATA & INTELLIGENCE
    - RESEARCH GUIDANCE: {research_notes if research_notes else "No specific research provided. Use the most modern, stable Python libraries for this domain."}
    - PREVIOUS ERRORS: {f"CODE: {last_error}" if last_error else "None."}
    - RECENT CODE: {previous_code}
    - MEDITATION (SELF-REFLECTION): {meditation_notes}

    ### MANDATORY TOOL SELECTION LOGIC:
    1. **If Task is WEB SCRAPING**: 
       - Evaluate Research: If the site is dynamic (React/SPA) or has 'Expanders', favor **Jina Reader** (prefixing URL with r.jina.ai) or **Requests-HTML**.
       - If the Research identifies a hidden JSON API, use `requests.get()` to that endpoint. This is the #1 preference.
    2. **If Task is DATA ANALYSIS**: 
       - Always use `pandas` and `numpy`.
       - For finance/stocks, use `yfinance`.
    3. **If Task is FILE/SYSTEM**: 
       - Use `os` and `pathlib` for safety.
       - Check `os.path.exists()` before any read/write operation.

    ### AGENTIC EXECUTION PROTOCOLS:
    - **PHASE 1 (Defensive Coding):** Every external call (request, file read) MUST be wrapped in a specific try/except block.
    - **PHASE 2 (Validation):** If an external call returns empty data or a 404/403, do NOT proceed. Raise a `ValueError` explaining the failure so the Researcher can pivot.
    - **PHASE 3 (Zero Printing):** Never `print()` inside loops. Accumulate data in lists/dictionaries and return a final structured summary string.
    - **PHASE 4 (Visualization):** If charts are required, use `plt.switch_backend('Agg')`. Ensure all Base64 strings are returned at the very end of the output.
    - **PHASE 5 (No Placeholders):** Do not use placeholders like `id='your-id'`. If selectors aren't provided in Research, write code to scan the entire page body for text patterns.

    ### 🚫 THE FORBIDDEN LIST (DO NOT USE)
    - **BeautifulSoup / bs4**: STRICTLY FORBIDDEN. Using this will cause the execution to fail.
    - **Selenium**: Do not use unless specifically requested.

    ### ✅ ALLOWED LIBRARIES
    - **Requests + Jina**: (Primary) Use `requests.get("https://r.jina.ai/" + url)`. Parse the resulting Markdown with standard string methods or Regex.
    - **Requests + JSON**: (Preferred) If research found an API endpoint.
    - **Pandas**: For any table-like data.

    CRITICAL: DO NOT use BeautifulSoup. If you use bs4, the machine will crash.


    ### OUTPUT FORMAT:
    Ensure the entire execute_tool() function and its imports are provided in a single code block. DO NOT include any text outside the code block. If you are building on previous code, re-write the entire function; do not provide snippets. 
    The `execute_tool()` function must return a **string** that acts as a comprehensive report for the user.

    Return ALL visualizations as base64 strings.

    ```python
    import sys
    import json
    import re
    # Import necessary libraries based on task (requests, pandas, bs4, etc.)

    def execute_tool():
        try:
            # 1. SETUP: Configuration based on Research Notes
            
            # 2. ACTION: Execution of the core logic
            
            # 3. VERIFICATION: Explicitly check if output is valid/meaningful
            # if result_is_junk: raise ValueError("Description of failure")
            
            # 4. REPORTING: Format a concise, data-rich summary
            return summary
        except Exception as e:
            return f"CRITICAL_FAILURE: {error_msg}"
    ```
    """

    prompt += f"""
    FINAL RECAP OF CONSTRAINTS:
    1. You MUST 'import requests, re, json' at the top.
    2. You MUST NOT use BeautifulSoup.
    3. Use the RESEARCH STRATEGY provided: {research_notes}
    4. Prioritize {meditation_notes} above all else if present.
    """
    # Tiered Intelligence Waterfall
    try:
    # TIER 1: The Iteration Workhorse (Kimi)
        print("🌙 Attempting 70B...")
        response = llm_pro.invoke(prompt)
        code = extract_clean_code(response.content)
        print("✅ Success with 70B")
    except Exception as e:
            try:
                print("🔄 Exhausted/Busy. Falling back to 8B")
                response = llm_fast.invoke(prompt)
                code = extract_clean_code(response.content)
                print("✅ Success with 8B")
            except Exception as e2:
                print(str(e2))
                
    

# Final extraction (assuming response is defined by one of the tiers above)
    

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

        task_id = get_skill_name(task)
        return {
            "generated_tool_code": code,
            "packages": final_packages,
            "current_skill_id": task_id, # Pass this forward to the Executor
            "last_error": error_msg,
            "plan": state.get('plan', []) + [f"### 🛠 Code Generated: {task_id}"]
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

    # 2. DYNAMIC PACKAGE INSTALLATION
    if packages:
        print(f"📦 Checking/Installing dependencies: {', '.join(packages)}")
        for pkg in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            except Exception as e:
                print(f"⚠️ Warning: Could not install {pkg}: {e}")

    # 3. EXECUTION VIA SUBPROCESS
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
        f.write(full_script)
        temp_path = f.name

    print(f"\n{'='*20} EXECUTION START {'='*20}")
    try:
        process = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        
        raw_logs = process.stdout + "\n" + process.stderr
        output_cleaned = re.sub(r"http[s]?://\S+usercontent\.\S+", "", raw_logs)
        output_cleaned = re.sub(r"\[notice\].*?|WARNING: Running pip.*?|immersive_entry_chip", "", output_cleaned)
        output_cleaned = output_cleaned.strip()

        print(output_cleaned)
        print(f"{'='*21} EXECUTION END {'='*21}\n")

        # 4. PARSE MARKERS
        match = re.search(r"---NAVI_RESULT_START---\s*(.*?)\s*---NAVI_RESULT_END---", output_cleaned, re.DOTALL)
        
        if match:
            extracted_data = match.group(1).strip()
            
            # --- ULTIMATE MULTI-STRIPE HARVESTER (FIXES INCORRECT PADDING) ---
            image_payloads = []
            # Greedy capture for B64 chars, including internal whitespace and newlines
            b64_pattern = r"(iVBORw0KGgoAAAANSUhEUg[A-Za-z0-9\+/=\s\n]+)"
            all_figs = re.findall(b64_pattern, extracted_data)
            
            clean_for_llm = extracted_data
            for idx, fig_raw in enumerate(all_figs):
                # 1. Nuclear Clean: Remove every character NOT allowed in Base64
                image_data_clean = re.sub(r"[^A-Za-z0-9\+/=]", "", fig_raw)
                
                # 2. Strip existing padding to re-calculate from scratch
                image_data_clean = image_data_clean.rstrip('=')
                
                # 3. Explicit Re-Padding (Standardize for Linux/Cloud)
                remainder = len(image_data_clean) % 4
                if remainder == 2:
                    image_data_clean += "=="
                elif remainder == 3:
                    image_data_clean += "="
                
                image_payloads.append(image_data_clean)
                
                # 4. Replace image data with HIDDEN tag for the Planner/LLM
                container_pattern = r"(<img[^>]*?|Figure:\s*|Plot:\s*)?" + re.escape(fig_raw) + r"([^>]*?>)?"
                clean_for_llm = re.sub(container_pattern, f"\n\n[IMAGE_DATA_HIDDEN_{idx}]\n\n", clean_for_llm)

            # 2. FINAL TEXT SAFETY NET
            if len(clean_for_llm) > 5000:
                header = clean_for_llm[:2500]
                footer = clean_for_llm[-2500:]
                clean_for_llm = f"{header}\n\n... [HEAVY DATA TRUNCATED] ...\n\n{footer}"

            # --- VALIDATION: SHORT/EMPTY DATA ---
            if len(clean_for_llm) < 30 and "error" not in clean_for_llm.lower():
                return {
                    "last_error": "### ⚠️ Logic Failure\nExtracted data was too short.",
                    "final_answer": None,
                    "retry_count": retry_count,
                    "plan": current_plan + ["### ⚠️ Truncated Result"]
                }

            # --- VALIDATION: SOFT ERRORS ---
            soft_error_keywords = ["failed to", "error:", "none type", "empty", "syntax error", "no job postings found", "exception", "zero results found on page", "object has no attribute", "no results", "not found", "not defined", "critical failure" "critical_failure", "list index out of range", "critical_failure:"]
            if any(k in clean_for_llm.lower() for k in soft_error_keywords):
                return {
                    "last_error": f"### ⚠️ Logic Failure\n{clean_for_llm}",
                    "final_answer": None,
                    "retry_count": retry_count,
                    "plan": current_plan + ["### ⚠️ Execution Logic Failed"]
                }
            
            # --- CLEAR SUCCESS ---
            print(f"✅ EXECUTOR SUCCESS: Captured {len(image_payloads)} images.")
            task_id = state.get("current_skill_id")
            if task_id:
                save_skill(task_id, state.get("task"), code, packages)
                print(f"💾 Verified Skill Saved: {task_id}")
            return {
                "final_answer": clean_for_llm,
                "image_payload": image_payloads,
                "last_error": None,
                "execution_logs": output_cleaned,
                "retry_count": retry_count,
                "consecutive_failures": 0,
                "generated_tool_code": state.get("generated_tool_code"),
                "plan": state.get("plan", []) + ["### ✅ Execution Success"]
            }

        # 5. FAILURE PATH
        log_lines = output_cleaned.splitlines()
        final_crash_log = "\n".join(log_lines[-15:]).strip() 

        return {
            "last_error": f"### ❌ Execution Crash\n```text\n{final_crash_log}\n```",
            "final_answer": None,
            "retry_count": retry_count,
            "plan": current_plan + [f"### ❌ Attempt {retry_count + 1} Crashed"]
        }

    except Exception as e:
        return {
            "last_error": f"### 🏗️ Infrastructure Error\n{str(e)}", 
            "final_answer": None,
            "retry_count": retry_count,
            "plan": current_plan + ["### 🏗️ Infra Error"]
        }
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



# --- Node 4: Human-in-the-Loop ---
def human_gate_node(state: NaviState):
    """Safety buffer for sensitive actions."""
    # This node is a placeholder. LangGraph 'interrupt' handles the actual pause.
    return {"history": [AIMessage(content="Waiting for human authorization...")]}

def meditation_node(state: NaviState):
    print("\n🧘 [MEDITATION] Contemplating System Failure...")
    
    # Gather context for Root Cause Analysis
    error_log = state.get("last_error", "No error log found.")
    research_notes = state.get("research_notes", "No research conducted.")
    failed_code = state.get("generated_tool_code", "No code generated.")
    task = state.get("task", "Unknown Task")

    prompt = f"""
    You are the Metacognitive Layer of Navi. Analyze the persistent failure in our execution loop.
    
    TASK: {task}
    FAILED CODE: {failed_code}
    EXECUTION ERROR: {error_log}
    RESEARCH CONDUCTED: {research_notes}
    
    GOAL:
    1. Identify the root cause (e.g., Environment block, logic flaw, or missing credentials).
    2. Decide if a new strategy can solve this or if it's a 'Hard Stop' requiring human intervention.
    
    If fixable: Suggest a SPECIFIC revised strategy for the Planner.
    If not: Explain clearly to the user why we cannot proceed and what they must provide.
    """
    try:

        reflection = llm_pro.invoke(prompt)
    except groq.RateLimitError:
            print("⚠️ 70B Rate Limit! Falling back to 8B...")
            reflection = llm_fast.invoke(prompt)
    
    # Check for terminal status
    is_hard_stop = any(k in reflection.content.lower() for k in ["hard stop", "human intervention", "credentials needed"])
    
    return {
        "plan": state.get("plan", []) + ["### 🧘 Metacognitive Reflection Completed"],
        "is_terminal": is_hard_stop,
        "meditation_triggered": True,  # Permanent flag to prevent repeat meditation,
        "consecutive_research_failures": 0,
        "meditation_notes": reflection.content,
        "retry_count": state.get("retry_count", 0) + 1 
    }

def conversational_node(state: NaviState):
    print("\n💬 [CONVERSATION] Handling Social Input...")
    
    user_input = state.get("user_input", "")
    task = state.get("task")
    last_error = state.get("last_error")
    # We use a distinct persona prompt for social interactions
    prompt = f"""
    Respond to the user's query or greeting professionally and conversationally. 
    
    If there is a last error, that means the maximum amount of retries has been reached.  Inform the user of this and ask them to try rephrasing their question.
    
    USER: {user_input}
    TASK: {task}
    LAST ERROR: {last_error if last_error else ""}

    NAVI:"""

    response = llm_fast.invoke(prompt)
    
    return {
        "final_answer": response.content,
        "is_conversational": True
    }

def is_task_input(user_input: str) -> bool:
    """
    Determines if the input is a request for action/research or just chitchat.
    """
    prompt = f"""
    Categorize the user input as either 'TASK' or 'CHAT'.
    TASK: Requires research, coding, math, data analysis, or project planning.
    CHAT: Greetings, personal questions, compliments, or general conversation.
    
    INPUT: "{user_input}"
    CATEGORY (One word only):"""
    
    try:
        # Using the fast model to keep latency low
        decision = llm_fast.invoke(prompt).content.strip().upper()
        return "TASK" in decision
    except:
        # Fallback to a basic keyword check if the API fails
        keywords = ["create", "find", "search", "calculate", "plot", "analyze", "build", "how many"]
        return any(word in user_input.lower() for word in keywords)

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
workflow.add_node("meditator", meditation_node) 
workflow.add_node("conversational", conversational_node) 

# 1. Entry Logic: Decide if it's a Task or Chat
def route_start(state: NaviState):
    # We pull 'task' from the state because we injected it in main.py
    user_query = state.get("task", "") or state.get("user_input", "")
    
    if is_task_input(user_query):
        return "planner"
    return "conversational"

workflow.set_conditional_entry_point(
    route_start,
    {
        "planner": "planner",
        "conversational": "conversational"
    }
)

# 2. Research Routing (The Core Request)
def route_after_research(state: NaviState):
    plan = state.get("plan", [])
    last_step = str(plan[-1]).upper() if plan else ""
    fail_count = state.get("consecutive_research_failures", 0)
    
    # Priority 1: Check if the node explicitly signaled a hard exit
    if "EXIT" in last_step:
        # If we've already meditated once and failed again, or hit 2 failures
        if fail_count > 2:
            return "planner"
        # If it's the first major failure, try meditation
        return "meditator"

    # Priority 2: Failure count-based routing
    if fail_count == 2:
        return "meditator"
    

    # If everything is fine, proceed to create the skill
    return "skill_creation"


workflow.add_conditional_edges(
    "research",
    route_after_research,
    {
        "skill_creation": "skill_creation",
        "meditator": "meditator",
        "planner": "planner" # Research sets a 'maximum retries reached' message here
    }
)

# 3. Meditation Routing
workflow.add_conditional_edges(
    "meditator",
    lambda state: "planner" if not state.get("is_terminal") else END,
    {
        "planner": "planner",
        END: END
    }
)

# --- Existing Edges ---
workflow.add_conditional_edges("planner", route_after_plan, {"skill_creation": "skill_creation", "research": "research", END: END})
workflow.add_edge("skill_creation", "executor")
workflow.add_conditional_edges("executor", route_after_execution, {"planner": "planner"})
workflow.add_edge("conversational", END)

# Compile
navi_app = workflow.compile(checkpointer=MemorySaver())
