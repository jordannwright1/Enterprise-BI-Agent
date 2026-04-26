import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
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
from langchain_huggingface import HuggingFaceEmbeddings
import ast
from core.database import get_skill, save_skill, init_db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

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

import time
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize the Embeddings Model (Local & Free)
# Ensure your Pinecone Index is set to 384 dimensions for this model
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("navi-memory")

def memory_retrieval_node(state: NaviState):
    """Pulls relevant past context into the current state."""
    # Using user_input for better semantic matching
    query_text = state.get("user_input") or state.get("task")
    if not query_text:
        return {"memory_context": ""}

    try:
        # Generate the vector
        query_vector = embeddings_model.embed_query(query_text)
        
        # Search Pinecone
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        
        # Filter by relevance score (0.7 is a good middle ground)
        past_facts = [res.metadata['text'] for res in results.matches if res.score > 0.7]
        
        context_string = "\n".join(past_facts) if past_facts else "No relevant memories found."
        return {"memory_context": context_string}
        
    except Exception as e:
        print(f"⚠️ Memory Retrieval Error: {e}")
        return {"memory_context": ""}

def save_memory_node(state: NaviState):
    """Commits the successful interaction to the Vector DB."""
    user_msg = state.get("user_input")
    ai_res = state.get("final_answer")
    
    # Validation: Don't save if there's no answer or it's an error message
    if not ai_res or "ERROR" in ai_res.upper():
        return {}

    try:
        text_to_save = f"User: {user_msg}\nNavi: {ai_res}"
        vector = embeddings_model.embed_query(text_to_save)
        
        # Generate a unique ID using the current Unix timestamp
        current_timestamp = int(time.time()) 
        unique_id = f"mem_{current_timestamp}"
        
        index.upsert(vectors=[{
            "id": unique_id,
            "values": vector, 
            "metadata": {
                "text": text_to_save,
                "timestamp": current_timestamp
            }
        }])
        print(f"💾 Memory Saved: {unique_id}")
    except Exception as e:
        print(f"⚠️ Memory Save Error: {e}")
        
    return {}

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

def universal_scraper(url, task_query, max_depth=1):
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse
    import json
    import re
    from playwright.sync_api import sync_playwright
    import os
    
    result = {"mode": "structured_blocks", "status": "initializing", "data": ""}
    
    # --- 1. THE "END THE ANNOYANCE" TARGET LOGIC ---
    target_count = None
    try:
        domain = urlparse(url).netloc.split('.')[-2].lower() if '.' in url else url.lower()
        pattern = rf"{re.escape(domain)}.*?(\d+)"
        match = re.search(pattern, task_query.lower(), re.DOTALL)
        
        if match:
            extracted = int(match.group(1))
            if extracted <= 3:
                context = task_query.lower()[max(0, match.start()-10) : match.end()+30]
                if any(w in context for w in ['article', 'story', 'item', 'result', 'top', 'first']):
                    target_count = extracted
            else:
                target_count = extracted
    except Exception as e:
        print(f"[DEBUG] Extraction error: {e}")

    if not target_count:
        potential_nums = [int(n) for n in re.findall(r'\d+', task_query) if int(n) > 1]
        target_count = potential_nums[0] if potential_nums else 3
    
    print(f"[DEBUG] URL: {url} | Domain: {domain} | Final Target Count: {target_count}")

    # 2. KEYWORD REFINEMENT
    raw_keywords = re.findall(r'\w+', task_query.lower())
    stop_words = {'go', 'to', 'and', 'find', 'the', 'for', 'each', 'its', 'page', 'extract', 'com', 'http', 'https', 'milestone'}
    keywords = [k for k in raw_keywords if k not in stop_words and len(k) > 2]
    
    def is_valid_url(u):
        try:
            res = urlparse(u)
            return all([res.scheme, res.netloc])
        except: return False

    # --- 3. DISCOVERY HELPER (NetNavi Autonomy) ---
    def discovery_navigator(page, soup, current_url):
        """Finds 'Next' or pagination links using both Playwright and BeautifulSoup."""
        found_nav_links = []
        
        # Scenario A: Search Engine Pagination (DuckDuckGo/Google)
        if any(x in current_url for x in ["duckduckgo.com", "google.com"]):
            nav_selectors = ["a#next", "a:has-text('Next')", "a[aria-label*='Next']"]
        # Scenario B: Individual Site Pagination
        else:
            nav_selectors = [
                "a:has-text('Next')", "a:has-text('>')", 
                "a[class*='pagination-next']", "a[class*='next']",
                "a[href*='page=']", "a:has-text('Older')"
            ]

        # 1. Try Playwright Selectors (Dynamic/JS)
        for selector in nav_selectors:
            try:
                elements = page.query_selector_all(selector)
                for el in elements:
                    href = el.get_attribute("href")
                    if href:
                        full_url = urljoin(current_url, href)
                        if is_valid_url(full_url) and full_url != current_url:
                            found_nav_links.append(full_url)
            except: continue

        # 2. Fallback to BeautifulSoup (Static HTML) - Incorporating 'soup'
        if soup and not found_nav_links:
            # Look for <a> tags containing 'next' in text or classes
            for a in soup.find_all('a', href=True):
                text = a.get_text().lower()
                classes = str(a.get('class', [])).lower()
                if any(x in text for x in ['next', 'older', '>']) or 'next' in classes:
                    full_url = urljoin(current_url, a['href'])
                    if is_valid_url(full_url) and full_url != current_url:
                        found_nav_links.append(full_url)

        return list(dict.fromkeys(found_nav_links)) # Unique URLs only

    try:
        target_url = url.strip()
        if not target_url.startswith('http'): target_url = 'https://' + target_url

        visited, to_visit, raw_storage, seen_titles = set(), [(target_url, 0)], [], set()
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(os.getcwd(), ".playwright_bins")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(user_agent='Mozilla/5.0')

            while to_visit and len(raw_storage) < target_count:
                entry = to_visit.pop(0)
                curr, depth = entry[0], entry[1]

                if not is_valid_url(curr) or curr in visited or depth > max_depth + 1: 
                    continue
                
                visited.add(curr)
                print(f"[NAVIGATING] {curr} (Depth: {depth})")

                try:
                    page.goto(curr, wait_until='domcontentloaded', timeout=15000)
                    page.wait_for_timeout(2000) 
                    soup = BeautifulSoup(page.content(), 'html.parser')
                    
                    if depth == 0 or any(x in curr for x in ["page=", "start=", "p="]):
                        # --- HUB PAGE LOGIC ---
                        found_links_on_this_page = 0
                        link_selectors = 'a[data-testid="result-title-a"]' if "duckduckgo.com" in curr else 'a'
                        
                        for link in (soup.select(link_selectors) if "duckduckgo.com" in curr else soup.find_all('a', href=True)):
                            if len(raw_storage) + found_links_on_this_page >= target_count: break
                                
                            full_url = urljoin(curr, link['href'])
                            if any(x in full_url.lower() for x in ['category/', 'tag/', 'cart', 'login', 'about', 'contact']):
                                continue
                                
                            title = (link.get('title') or link.get_text() or "").strip()
                            if not title or len(title) < 5: continue
                            
                            score = 0
                            parent_check = link.find_parent(['article', 'h1', 'h2', 'h3', 'h4', 'tr'])
                            is_headline_class = any(x in str(link.get('class', [])).lower() for x in ["adventure", "title", "entry"])
                            
                            if parent_check or is_headline_class or len(title) > 35: score += 30
                            if any(k in title.lower() for k in keywords): score += 20
                            
                            if is_valid_url(full_url) and title not in seen_titles and score >= 30:
                                to_visit.append((full_url, 1))
                                seen_titles.add(title)
                                found_links_on_this_page += 1

                        # --- PAGINATION DISCOVERY ---
                        if len(raw_storage) + len(to_visit) < target_count:
                            # PASSING 'soup' HERE AS WELL
                            next_pages = discovery_navigator(page, soup, curr)
                            for next_url in next_pages:
                                if next_url not in visited:
                                    print(f"  [DISCOVERY] Found pagination: {next_url}")
                                    to_visit.append((next_url, 0))

                    else:
                        # --- DETAIL PAGE LOGIC ---
                        item_data = {}
                        author_tag = soup.find(['a', 'div', 'span'], class_=re.compile(r'author|byline|user|hnuser', re.I))
                        if author_tag: item_data['Source/Author'] = author_tag.get_text(strip=True)

                        if "news.ycombinator.com" in url:
                            comment = soup.find('span', class_='commtext')
                            if comment: item_data['Top Comment'] = comment.get_text(strip=True)[:400]
                        
                        p_tags = soup.find_all('p')
                        if p_tags:
                            longest_p = max(p_tags, key=lambda p: len(p.get_text()))
                            if len(longest_p.get_text()) > 50:
                                item_data['Summary'] = longest_p.get_text(strip=True)[:400]

                        h1 = soup.find('h1') or soup.find('title')
                        if item_data:
                            name = h1.get_text(strip=True) if h1 else "Item"
                            print(f"  [SUCCESS] Extracted: {name}")
                            raw_storage.append({"title": name, "details": item_data})

                except Exception as e:
                    print(f"  [SKIP] {curr}: {str(e)[:40]}")
                    continue
            
            browser.close()

        if not raw_storage:
            result["data"] = "No matching items found."
        else:
            table = [f"### Results for {urlparse(url).netloc}\n| Title | Extracted Details |", "| :--- | :--- |"]
            for entry in raw_storage[:target_count]:
                details = " <br> ".join([f"**{k}**: {v}" for k, v in entry['details'].items()])
                table.append(f"| {entry['title']} | {details} |")
            result["data"] = "\n".join(table)
            result["status"] = "success"

        return result
    except Exception as e:
        return {"mode": "error", "data": str(e)}
                

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
import os

def ensure_packages(package_list):
    """Installs missing packages safely and handles Playwright's binary requirements."""
    failed_packages = []
    
    for package in package_list:
        # Normalize the name for the __import__ check (e.g., 'beautifulsoup4' -> 'bs4')
        # Note: This check is a bit naive for some packages, but works for most.
        import_name = package.replace('-', '_')
        if package.lower() == "beautifulsoup4": import_name = "bs4"
        if package.lower() == "scikit-learn": import_name = "sklearn"
        if package.lower() == "pyyaml": import_name = "yaml"

        try:
            # Check if package is already available
            __import__(import_name)
        except ImportError:
            try:
                print(f"📦 System: Installing missing dependency: {package}...")
                # Standard pip install
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                
                # --- THE PLAYWRIGHT SPECIAL CASE ---
                if package.lower() == "playwright":
                    print("🚀 Forcing Playwright Browser into local project folder...")
                    # This path is relative to your root /mount/src/enterprise-bi-agent/
                    local_browser_path = os.path.join(os.getcwd(), ".playwright_bins")
                    os.makedirs(local_browser_path, exist_ok=True)
    
                    env = os.environ.copy()
                    env["PLAYWRIGHT_BROWSERS_PATH"] = local_browser_path
    
                    subprocess.check_call(
                    [sys.executable, "-m", "playwright", "install", "chromium"], 
                    env=env
                )

            except subprocess.CalledProcessError as e:
                print(f"❌ System: Failed to install {package} (Exit code: {e.returncode})")
                failed_packages.append(package)
            except subprocess.TimeoutExpired:
                print(f"⚠️ System: Playwright install timed out.")
                failed_packages.append(f"{package} (timeout)")
            except Exception as e:
                print(f"⚠️ System: Unexpected error installing {package}: {e}")
                failed_packages.append(package)
                
    return failed_packages


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


# --- Node 1: Planner ---
def planner_node(state: NaviState):
    """
    UNIVERSAL PLANNER NODE - V2 (Anti-Loop Edition)
    Acts as the central nervous system for routing between 
    Research, Execution, and Final Synthesis.
    """
    final_ans_raw = state.get("final_answer")
    last_error = state.get("last_error")
    retry_count = state.get("retry_count", 0)
    plan = state.get("plan", [])
    task = state.get("task")
    meditation_notes = state.get("meditation_notes")
    
    # ---------------------------------------------------------
    # 1. ESCALATION & TERMINATION (The "Panic" Logic)
    # ---------------------------------------------------------
    if meditation_notes or (retry_count >= 3 and last_error):
        if retry_count >= 3:
            print("🛑 Planner: Maximum recovery attempts reached. Terminating.")
            return {
                "is_terminal": True,
                "plan": plan + ["### 🏁 ACTION: TERMINATED"]
            }
        
        print(f"🧘 Planner: Meditation active. Routing to Recovery Mode ({retry_count + 1}/3)")
        return {
            "plan": plan + ["### 🛠️ ACTION: CODE"],
            "retry_count": retry_count + 1,
            "last_error": last_error 
        }

    # ---------------------------------------------------------
    # 2. AUDIT & RECURSION GATE (The "Is it Finished?" Logic)
    # ---------------------------------------------------------
    if final_ans_raw and not last_error:
        if plan and "EXIT" in str(plan[-1]).upper():
            return {}

        print(f"🧐 Planner: Auditing objective satisfaction (Cycle {retry_count})...")
        
        audit_prompt = f"""
        Objective: {task}
        Latest Result Data: {str(final_ans_raw)[:3000]}
        Current Memories of Past Conversations: {state['memory_context']}\n\nUse this context to inform your response if relevant.
        
        CRITICAL AUDIT:
        1. Does the result contain actual data points requested (e.g., names, stats, summaries)?
        2. Is the content mostly 'bot detected' errors or empty blocks?
        3. Is there a clear, high-value 'Next Step' that hasn't been taken?

        If the objective is substantially satisfied, or we have hit a point of diminishing returns, respond: 'COMPLETE'.
        If the data is genuinely missing or the results are only meta-links, respond: 'CONTINUE'.
        """
        audit_decision = llm_fast.invoke(audit_prompt).content.strip().upper()

        # LOOP BREAKER: If Auditor says CONTINUE but we've already tried to 
        # refine results 2+ times, we override and force COMPLETE to prevent loops.
        if "CONTINUE" in audit_decision and retry_count < 2:
            print("🔄 Planner: Data gap detected. Refreshing context for next cycle.")
            return {
                "plan": plan + ["### 🛠️ ACTION: CODE"], 
                "final_answer": None, 
                "last_error": None,
                "retry_count": retry_count + 1,
                "research_notes": f"Previous Discovery (Needs Refinement): {str(final_ans_raw)[:1000]}"
            }

        # --- FINAL SYNTHESIS ---
        print("✨ Planner: Logic satisfied. Synthesizing final response.")
        
        format_prompt = f"""
        Original User Task: {task}
        Verified Scrape Data: {final_ans_raw}

        Current Memories of Past Conversations: {state['memory_context']}\n\nUse this context to inform your response if relevant.

    
        You are a clean-up agent. Generate a professional response:
        1. Ignore any obviously hallucinated or placeholder data (e.g. news from 2024 if the scrape was for 2026).
        2. Use Markdown tables for comparisons and bullets for lists.
        3. If some data was blocked by bots, summarize what WAS found successfully.
        4. Remove all internal logs, [DEBUG] tags, or technical metadata.
        """
        summary_response = llm_fast.invoke(format_prompt).content.strip()
    
        return {
            "final_answer": summary_response, 
            "last_error": None,               
            "plan": plan + ["### 🏁 ACTION: EXIT"]
        }

    # ---------------------------------------------------------
    # 3. DYNAMIC ERROR RECOVERY
    # ---------------------------------------------------------
    if last_error:
        if retry_count < 2:
            print(f"🛠️ Planner: Execution error. Attempting Auto-Repair ({retry_count + 1})")
            return {
                "plan": plan + ["### 🛠️ ACTION: CODE"],
                "last_error": last_error,
                "retry_count": retry_count + 1
            }
        else:
            print("🔍 Planner: Code repair failed. Escalating to deep Research.")
            return {
                "plan": plan + ["### 🔍 ACTION: RESEARCH"],
                "last_error": last_error, 
                "retry_count": 0 
            }

    # ---------------------------------------------------------
    # 4. SKILL CACHE & COLD START
    # ---------------------------------------------------------
    skill_id = get_skill_name(task)
    existing_skill = get_skill(skill_id) 
    
    if existing_skill and not plan:
        print(f"🧠 Planner: Found existing Skill Match: {skill_id}")
        return {
            "plan": ["### 💾 ACTION: LOAD_SKILL"],
            "generated_tool_code": existing_skill['code'],
            "packages": existing_skill.get('packages', []),
            "retry_count": 0
        }

    # Default Entrance
    print("🚀 Planner: Initiating new Task Sequence.")
    return {
        "plan": ["### 🛠️ ACTION: CODE"], 
        "retry_count": 0
    }


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

    Current Memories of Past Conversations: {state['memory_context']}\n\nUse this context to inform your response if relevant.
    
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
        "diagnosis": "Short paragraph summary of the error and the solution to the problem.  You must include a solution that is one to two sentences long.",
        "solution_logic": "Two sentence strategy",
        "code": "The corrected python code block"
    }}
    """
    
    try:
        raw_response = llm_pro.invoke(reasoning_prompt).content.strip()
    except Exception as e:
        print(f"⚠️ 70B failed: {e}. Falling back to 8B...")
        try:
            # FALLBACK 1: Llama 8B
            raw_response = llm_fast.invoke(reasoning_prompt).content.strip()
        except (groq.RateLimitError, Exception):
            print("⚠️ 8B Rate Limit Error")
            

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
    research_notes = state.get("research_notes", "")
    meditation_notes = state.get("meditation_notes", "")
    previous_code = state.get("generated_tool_code", "")
    aggregated_research = state.get("aggregated_research", "")
    executor_logs = state.get("executor_logs", "") 
    error_msg = ""
    history = state.get("history", [])
    
    # Get the last message's content
    if not history:
        user_input = state.get("task", "") 
    else:
        user_input = history[-1].content

    # 1. Scraper Execution
    is_scraping_task = any(word in user_input.lower() for word in ["http", "scrape", "extract", "visit"])
    # 1. Identify ALL URLs in the prompt
    # This regex finds all unique links in the input
    urls = list(dict.fromkeys(re.findall(r'https?://[^\s/$.?#].[^\s]*', user_input)))
    
    if is_scraping_task and urls:
        new_research_blocks = []
        
        # 2. Iterate through every URL found
        for target_url in urls:
            print(f"🌐 [BATCH] Starting scrape for: {target_url}")
            
            # Run the scraper for the specific URL
            # Note: task_query=user_input allows the scraper to see the context for numbers
            raw_data = universal_scraper(target_url, task_query=user_input)
            
            if raw_data.get("status") == "success":
                # Wrap the data in a header for the LLM
                block = f"\n\n### DATA FROM {target_url}\n" + raw_data['data']
                new_research_blocks.append(block)
            else:
                print(f"⚠️ [BATCH] Failed to scrape {target_url}")

        # 3. Update the state with EVERYTHING we found
        if new_research_blocks:
            final_data = aggregated_research + "".join(new_research_blocks)
            
            # Create the final tool code with the full combined dataset
            generated_code = f'def execute_tool():\n    return """{final_data}"""'
            
            # We use the first URL to generate the skill ID (or a combined one)
            task_id = get_skill_name(f"multi_scrape_{len(urls)}")
            
            return {
                "generated_tool_code": generated_code,
                "aggregated_research": final_data,
                "current_skill_id": task_id,
                "last_error": None,
                "plan": plan + [f"### ✅ Scraped {len(new_research_blocks)} sources successfully."]
            }
            
        # --- PATH B: THE DIAGNOSTIC FAILURE ---
        else:
                print(f"⚠️ [DIAGNOSIS] Scraper failed. Passing logs to Architect for repair...")
                # Inject the failure logs into research notes so the LLM knows why it's fixing it
                failure_msg = raw_data.get('data', 'Unknown scraper error')
                research_notes += f"\n\n[INTERNAL SCRAPER FAILURE LOGS]:\n{failure_msg}"


    # 2. Final Prompt Construction
    prompt = f"""
    ### ROLE: Navi Automation Architect
    
    ### MISSION: {task}

    ### THE TOOLS (ONLY USE THESE)
    1. **To Search**: Use `from ddgs import DDGS`. 
       - Dictionary key for the link is 'href'.
    2. **To Scrape**: Use `universal_scraper(url, task_query)`. 
       - This function is ALREADY defined and injected. DO NOT import it.
       - **NEW CAPABILITY**: This tool now automatically handles pagination and link harvesting.

    ### THE RULES
    - Return ONLY the code for `execute_tool()`.
    - DO NOT use requests, BeautifulSoup, or playwright (these are handled internally by the scraper).
    - DO NOT explain anything.
    - Search using site:domain.com "query" to ensure accuracy.

    ### EXAMPLE STRUCTURE
    ```python
    from ddgs import DDGS

    user_goal = state.get("task")

    def execute_tool():
        with DDGS() as ddgs:
            # Step 1: Search
            links = [r['href'] for r in ddgs.text("query", max_results=5)]
        
        # Step 2: Scrape each link. The recursive logic is handled within universal_scraper.
        return [universal_scraper(url, task_query=user_goal) for url in links]
    ```
    
    ### GENERATE CODE:
    """

    # Tiered Intelligence Waterfall
    try:
        print("🌙 Attempting 70B (Skill Creator)...")
        response = llm_pro.invoke(prompt)
        code = extract_clean_code(response.content)
        print("✅ Success with 70B")
    except Exception as e:
        print(f"🔄 70B Exhausted/Busy: {e}. Falling back to 8B...")
        response = llm_fast.invoke(prompt)
        code = extract_clean_code(response.content)
        print("✅ Success with 8B")

    if not code:
        return {"last_error": "### ❌ Error\nNo Python code found."}

    try:
        ast.parse(code)
        final_packages = extract_dependencies(code)

        if final_packages:
            failed_installs = ensure_packages(final_packages)
            if failed_installs:
                return {
                    "last_error": f"### ❌ Dependency Error\nCould not install: {failed_installs}.",
                    "plan": plan + [f"### ⚠️ Install Failed: {failed_installs}"]
                }

        task_id = get_skill_name(task)
        try:
            print(f"DEBUGGING CODE TO BE PARSED:\n{code}") # Look at this in the Streamlit terminal!
            ast.parse(code)
        except SyntaxError as e:
            print(f"SYNTAX ERROR DETECTED: {e}")
        return {
            "generated_tool_code": code,
            "packages": final_packages,
            "current_skill_id": task_id,
            "last_error": last_error,
            "plan": plan + [f"### 🛠 Code Generated: {task_id}"]
        }
    except SyntaxError as e:
        return {"last_error": f"### ❌ Syntax Error\n{str(e)}", "plan": plan + ["### ❌ Syntax Error"]}
    except Exception as e:
        return {"last_error": f"### ❌ Creator Error\n{str(e)}", "plan": plan + ["### ❌ Creator Node Error"]}
                


# --- Node 3: Executor ---
import textwrap
import tempfile
import subprocess
import os
import re
import sys

def executor_node(state: NaviState):
    print("\n🚀 [SUBPROCESS] Starting Lite Execution...")
    
    code = state.get('generated_tool_code')
    packages = state.get('packages', []) 
    retry_count = state.get('retry_count', 0)
    current_plan = state.get('plan', [])
    task = state.get("task")
    internal_tools = ["universal_scraper", "ddgs_search", "NaviState"] 
    filtered_packages = [p for p in packages if p not in internal_tools]
    if not code:
        return {
            "last_error": "### ❌ Execution Failed\nNo code found.",
            "retry_count": retry_count + 1,
            "plan": current_plan + ["### ❌ No Code to Execute"]
        }
    import inspect
    try:
        scraper_code = inspect.getsource(universal_scraper)
    except Exception as e:
        print(f"⚠️ Warning: Could not find universal_scraper source: {e}")
        scraper_code = "def universal_scraper(url): return 'Error: Scraper source missing.'"
    # 1. WRAP & DEDENT (Ensuring strict output markers)
    raw_script = f"""
import sys
import io
import json
import re
import os
import base64
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Force UTF-8 and unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# --- INJECTED UTILITIES (The Recursive universal_scraper) ---
{scraper_code}

# --- NAVI GENERATED CODE ---
{code}

if __name__ == "__main__":
    try:
        # Check if execute_tool exists
        if 'execute_tool' not in globals():
            sys.stdout.write("EXECUTION_ERROR: Function 'execute_tool' not defined in generated code.\\n")
        else:
            # Call the generated tool. It has access to the recursive scraper logic.
            result = execute_tool()
            output = str(result) if result is not None else "NAVI_EMPTY_RESULT"
            
            # Standardized output markers for the graph parser
            sys.stdout.write("\\n---NAVI_RESULT_START---\\n")
            sys.stdout.write(output)
            sys.stdout.write("\\n---NAVI_RESULT_END---\\n")
            sys.stdout.flush()
    except Exception as e:
        # Pass the error back to the graph so the 'meditator' or 'planner' can debug
        sys.stdout.write(f"EXECUTION_ERROR: {{e}}\\n")
        sys.stdout.flush()
"""

    full_script = textwrap.dedent(raw_script).strip()

    # 2. DYNAMIC PACKAGE INSTALLATION
    if filtered_packages:
        print(f"📦 Checking/Installing dependencies: {', '.join(filtered_packages)}")
        for pkg in filtered_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            except Exception as e:
                print(f"⚠️ Warning: Could not install {pkg}: {e}")

    # 3. EXECUTION VIA SUBPROCESS
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
            f.write(full_script)
            temp_path = f.name

        print(f"\n{'='*20} EXECUTION START {'='*20}")
        # Force the environment variables for Playwright here too
        env_vars = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        if "PLAYWRIGHT_BROWSERS_PATH" not in env_vars:
            env_vars["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(os.getcwd(), ".playwright_bins")

        process = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=120, # Increased for slow scrapers
            env=env_vars
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
            
            # --- ULTIMATE MULTI-STRIPE HARVESTER ---
            image_payloads = []
            b64_pattern = r"(iVBORw0KGgoAAAANSUhEUg[A-Za-z0-9\+/=\s\n]+)"
            all_figs = re.findall(b64_pattern, extracted_data)
            
            clean_for_llm = extracted_data
            for idx, fig_raw in enumerate(all_figs):
                image_data_clean = re.sub(r"[^A-Za-z0-9\+/=]", "", fig_raw).rstrip('=')
                remainder = len(image_data_clean) % 4
                if remainder == 2: image_data_clean += "=="
                elif remainder == 3: image_data_clean += "="
                
                image_payloads.append(image_data_clean)
                
                container_pattern = r"(<img[^>]*?|Figure:\s*|Plot:\s*)?" + re.escape(fig_raw) + r"([^>]*?>)?"
                clean_for_llm = re.sub(container_pattern, f"\n\n[IMAGE_DATA_HIDDEN_{idx}]\n\n", clean_for_llm)

            # TOKEN SAFETY
            if len(clean_for_llm) > 5000:
                clean_for_llm = f"{clean_for_llm[:2500]}\n\n... [TRUNCATED] ...\n\n{clean_for_llm[-2500:]}"

            # --- UPDATED VALIDATION: DIFFERENTIATING CONTENT FROM CRASHES ---
            # We only trigger a "Logic Failure" if the output is extremely short AND contains an error keyword.
            # This allows articles about "market failures" or "system errors" to pass.
            python_errors = ["syntaxerror", "traceback", "execution_error", "critical_failure", "modulenotfounderror"]
            is_python_crash = any(k in output_cleaned.lower() for k in python_errors)
            is_empty_or_tiny = len(clean_for_llm) < 50 or "NAVI_EMPTY_RESULT" in clean_for_llm
            
            if is_python_crash or (is_empty_or_tiny and "error" in clean_for_llm.lower()):
                return {
                    "last_error": f"### ⚠️ Logic Failure\n{clean_for_llm if not is_empty_or_tiny else 'Result was empty or returned a crash log.'}",
                    "executor_logs": output_cleaned,
                    "final_answer": None,
                    "retry_count": retry_count,
                    "plan": current_plan + ["### ⚠️ Execution Logic Failed"]
                }
            
            # --- CLEAR SUCCESS: SAVE TO DB ---
            task_id = state.get("current_skill_id")
            if task_id:
                # Assuming save_skill is imported or available in scope
                try:
                    save_skill(task_id, task, code, packages)
                    print(f"💾 Verified Skill Saved: {task_id}")
                except NameError:
                    print("⚠️ save_skill not defined, skipping DB save.")

            return {
                "final_answer": clean_for_llm,
                "image_payload": image_payloads,
                "last_error": None,
                "executor_logs": output_cleaned,
                "retry_count": retry_count,
                "plan": current_plan + ["### ✅ Execution Success"]
            }

        # 5. FAILURE PATH (NO MARKERS)
        log_lines = output_cleaned.splitlines()
        final_crash_log = "\n".join(log_lines[-15:]).strip() 

        return {
            "last_error": f"### ❌ Execution Crash\n{final_crash_log}",
            "executor_logs": output_cleaned,
            "final_answer": None,
            "retry_count": retry_count + 1,
            "plan": current_plan + [f"### ❌ Attempt {retry_count + 1} Crashed"]
        }

    except Exception as e:
        return {
            "last_error": f"### 🏗️ Infrastructure Error\n{str(e)}", 
            "executor_logs": str(e),
            "final_answer": None,
            "retry_count": retry_count,
            "plan": current_plan + ["### 🏗️ Infra Error"]
        }
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass



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
    If not: Explain clearly to the user why we cannot proceed and what they must provide.  If you decide escalation to human intervention is necessary be sure to include the words 'hard stop' or 'human intervention' in your response.
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
    is_terminal = state.get("is_terminal", False)
    meditation_notes = state.get("meditation_notes", "")
    last_error = state.get("last_error", "")
    history = state.get("history")
    if is_terminal:
        prompt = f"""
        The system has reached a Hard Stop. You must inform the user why the task cannot be completed.
        
        REASON FROM MEDITATOR: {meditation_notes}
        TECHNICAL ERROR: {last_error}

        Current Memories of Past Conversations: {state['memory_context']}\n\nUse this context to inform your response if relevant.
        
        INSTRUCTIONS:
        - Be professional but firm.
        - Explain exactly what the user needs to provide (e.g., API keys, better URL, etc.).
        - Do not attempt to solve the task further.
        """
    else:
        prompt = f"""
        Respond to the user's query or greeting professionally and conversationally. 

        Current Memories of Past Conversations (Long Term Memory): {state['memory_context']}\n\nUse this context to inform your response if relevant.

        Short Term Memory: {history}
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
    # If no plan exists, we default to skill_creation to try and make progress
    if not plan: 
        print("🎯 Router: No plan found. Defaulting to Skill Creation.")
        return "skill_creation"
    
    last_step = str(plan[-1]).upper()
    print(f"DEBUG [Router] - Deciding path for: {last_step}")

    # --- EXIT CONDITION ---
    # The Planner must explicitly say 'EXIT', 'TERMINATED', or 'FINISH' 
    # when it sees that the results in history satisfy the user's original request.
    if any(keyword in last_step for keyword in ["EXIT", "TERMINATED", "FINISH", "COMPLETED"]):
        print("🏁 Router: Task marked as complete. Terminating workflow.")
        return END
    
    # --- RESEARCH BRANCH ---
    # If the Planner realizes it needs more external info (like searching for links)
    if "RESEARCH" in last_step:
        print("🎯 Router: Match! Redirecting to Research Node.") 
        return "research"
    
    # --- ACTION BRANCH ---
    # Default to skill_creation to execute the next tool in the sequence
    print("🛠️ Router: Proceeding to Skill Creation/Execution.")
    return "skill_creation"

def route_after_research(state: NaviState):
    """
    Decides the path after a research attempt.
    Routes to Skill Creation on success, or Meditation on failure.
    """
    plan = state.get("plan", [])
    last_step = str(plan[-1]).upper() if plan else ""
    fail_count = state.get("consecutive_research_failures", 0)
    meditation_used = state.get("meditation_triggered", False)
    
    # Priority 1: Check for explicit EXIT signal from the Researcher
    # This usually means the LLM couldn't find relevant info.
    if "EXIT" in last_step:
        # If we already meditated or have failed too many times, 
        # go to Planner to finalize a "failure report" and Save Memory.
        if fail_count >= 2 or meditation_used:
            print("🛑 Route: Research failed twice. Escalating to Planner for final report.")
            return "planner"
        
        # If this is the first major wall, try to Meditate/Refocus
        print("🧘 Route: Research hit a wall. Triggering Meditation.")
        return "meditator"

    # Priority 2: Hard failure cap
    # Even if no EXIT was called, if we've looped 3 times, stop the madness.
    if fail_count >= 3:
        print("🛑 Route: Maximum research attempts reached. Exiting to Planner.")
        return "planner"

    # Priority 3: Success Path
    # If the research yielded results and didn't trigger an error, build the tool.
    print("🛠️ Route: Research successful. Moving to Skill Creation.")
    return "skill_creation"


def route_after_execution(state: NaviState):
    last_err = state.get("last_error")
    # We check the final_answer which holds the string result of the subprocess
    raw_ans = str(state.get("final_answer", "")).lower()
    
    # 1. Error Handling: If the code crashed or the soft-check caught an error
    if last_err or "error occurred" in raw_ans or "traceback" in raw_ans:
        print(f"🎯 Router: Execution issue detected. Redirecting to Planner for repair.")
        return "planner"
    
    # 2. Success Path: Always loop back to the Planner.
    # This is crucial for multi-step tasks. After searching, Navi returns here, 
    # goes to the Planner, sees the search results, and then plans the 'Scrape' step.
    print(f"✅ Router: Execution successful. Returning to Planner for next instruction.")
    return "planner"

# --- Node Configuration ---
workflow = StateGraph(NaviState)

# --- Add the Memory Nodes ---
workflow.add_node("memory_recall", memory_retrieval_node) # PULL
workflow.add_node("memory_save", save_memory_node)        # PUSH

# --- Existing Nodes ---
workflow.add_node("planner", planner_node)
workflow.add_node("skill_creation", skill_creator_node)
workflow.add_node("executor", executor_node)
workflow.add_node("research", research_node)
workflow.add_node("meditator", meditation_node) 
workflow.add_node("conversational", conversational_node) 

# --- 1. NEW ENTRY POINT: Memory First ---
# We want Navi to remember BEFORE she thinks.
workflow.set_entry_point("memory_recall")

# After recall, we check if it's a task or a chat
def route_after_memory(state: NaviState):
    user_query = state.get("task", "") or state.get("user_input", "")
    if is_task_input(user_query):
        return "planner"
    return "conversational"

workflow.add_conditional_edges(
    "memory_recall",
    route_after_memory,
    {
        "planner": "planner",
        "conversational": "conversational"
    }
)

# --- 2. Research & Meditation (Your existing logic) ---
workflow.add_conditional_edges(
    "research",
    route_after_research,
    {
        "skill_creation": "skill_creation",
        "meditator": "meditator",
        "planner": "planner" 
    }
)

workflow.add_conditional_edges(
    "meditator",
    lambda state: "planner" if not state.get("is_terminal") else "memory_save", # Route to save before end
    {
        "planner": "planner",
        "memory_save": "memory_save"
    }
)

# --- 3. UPDATED EXIT LOGIC: Save before END ---
# Replace END with "memory_save" in all final steps

def route_after_plan(state: NaviState):
    plan = state.get("plan", [])
    if plan and "EXIT" in str(plan[-1]).upper():
        return "memory_save" # Redirect from END to memory_save
    # ... your existing logic for research/skill_creation
    return "skill_creation" # placeholder

workflow.add_conditional_edges(
    "planner", 
    route_after_plan, 
    {
        "skill_creation": "skill_creation", 
        "research": "research", 
        "memory_save": "memory_save" # Now points here instead of END
    }
)

workflow.add_edge("skill_creation", "executor")
workflow.add_conditional_edges("executor", route_after_execution, {"planner": "planner"})

# Conversational also needs to remember what it said
workflow.add_edge("conversational", "memory_save")

# --- 4. THE FINAL TERMINATION ---
workflow.add_edge("memory_save", END)

# Compile
navi_app = workflow.compile(checkpointer=MemorySaver())
