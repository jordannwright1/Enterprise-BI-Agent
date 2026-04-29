import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
import os
import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from core.state import NaviState, NaviEngine
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
        
import asyncio
import inspect
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

def run_sync_scraper(fn, *args, **kwargs):
    nest_asyncio.apply()
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Generate the coroutine if it's an async function
    if inspect.iscoroutinefunction(fn):
        coro = fn(*args, **kwargs)
    elif inspect.iscoroutine(fn):
        coro = fn
    else:
        return fn(*args, **kwargs)

    # --- THE STREAMLIT FIX ---
    if loop.is_running():
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    
    return loop.run_until_complete(coro)


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

base_pw_path = "/mount/src/bi-agent-v2/pw-browsers" if os.path.exists("/mount/src/bi-agent-v2") else os.path.join(os.getcwd(), "pw-browsers")

os.environ["PLAYWRIGHT_BROWSERS_PATH"] = base_pw_path
import platform
def get_executable_path():
    if platform.system() == "Darwin":  # Mac
        return os.path.join(base_pw_path, "chromium_headless_shell-1217/chrome-headless-shell-mac-arm64/chrome-headless-shell")
    else:  # Linux (Streamlit)
        return os.path.join(base_pw_path, "chromium_headless_shell-1217/chrome-headless-shell-linux64/chrome-headless-shell")


import os
import platform
import asyncio
import json
import re
import traceback
from collections import Counter

# --- HELPER: CROSS-PLATFORM PATHS ---
base_pw_path = "/mount/src/bi-agent-v2/pw-browsers" if os.path.exists("/mount/src/bi-agent-v2") else os.path.join(os.getcwd(), "pw-browsers")

def get_executable_path():
    """Returns the correct binary path based on the operating system."""
    if platform.system() == "Darwin":  # Mac
        return os.path.join(base_pw_path, "chromium_headless_shell-1217/chrome-headless-shell-mac-arm64/chrome-headless-shell")
    else:  # Linux (Streamlit)
        return os.path.join(base_pw_path, "chromium_headless_shell-1217/chrome-headless-shell-linux64/chrome-headless-shell")

# --- MAIN ASYNC FUNCTION ---
async def universal_scraper(url, task_query, max_depth=1, fields=None, label_context=None):
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright
    
    result = {"mode": "structured_blocks", "status": "initializing", "data": ""}
    
    # Ensure Playwright knows where our custom install folder is
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = base_pw_path
    
    try:
        target_url = url.strip()
        if not target_url.startswith('http'): 
            target_url = 'https://' + target_url
        
        async with async_playwright() as p:
            # Launch with all necessary flags for Streamlit stability
            browser = await p.chromium.launch(
            executable_path=get_executable_path(),
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",  # Redirects memory to /tmp (The most important flag)
                "--disable-gpu",            # Prevents GPU process crashes
                "--single-process",         # Forces all browser logic into the main app process
                "--no-zygote",              # Stops the browser from spawning a "helper" process
                "--disable-extensions",      # Minimizes overhead
                "--proxy-server='direct://'", # Skips proxy lookups
                "--proxy-bypass-list=*"
            ]
        )
            
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            print(f"[RECON] Landing: {target_url}")
            await page.goto(target_url, wait_until='networkidle', timeout=30000)
            
            # --- PHASE 0: INCREMENTAL SCROLL ---
            print("[SCROLL] Activating lazy-loaded content...")
            for _ in range(5):
                await page.evaluate("window.scrollBy(0, 800)")
                await page.wait_for_timeout(1200)
            
            raw_data, seen = [], set()
            current_depth = 0
            
            # Universal Noise & Price Regex
            noise_rex = re.compile(r'latest news|most popular|trending|newsletter|subscribe|follow us|privacy policy|terms|copyright|sign in|navigation|view all|sponsored|protection plan|recently viewed', re.I)
            price_rex = re.compile(r'\$\d+(?:\.\d{2})?')

            while current_depth <= max_depth:
                await page.wait_for_timeout(2000)
                html_content = await page.content()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Clean structural fluff
                for nt in soup.find_all(['nav', 'footer', 'aside', 'script', 'style']): 
                    nt.decompose()

                # --- PHASE 1: TEXT-INWARD SCANNING ---
                text_node_scores = []
                task_keywords = [w.lower() for w in task_query.split() if len(w) > 3]

                for tag in soup.find_all(True):
                    text = tag.get_text(" ", strip=True)
                    if not text or len(text) < 15: continue
                    if noise_rex.search(text): continue

                    score = (len(text) // 40) * 5 
                    score += sum(25 for k in task_keywords if k in text.lower())
                    if tag.name in ['h1', 'h2', 'h3', 'h4']: score += 20
                    
                    if score > 10:
                        text_node_scores.append({"tag": tag, "score": score})

                # --- PHASE 2: BUBBLE-UP ---
                parent_class_counter = Counter()
                for entry in text_node_scores:
                    curr = entry['tag']
                    for _ in range(5):
                        if curr.parent:
                            curr = curr.parent
                            classes = curr.get('class', [])
                            if classes:
                                c_str = " ".join(classes).lower()
                                if any(x in c_str for x in ['carousel', 'hero', 'slider', 'featured', 'ad-']):
                                    continue
                                weight = 1
                                if any(x in c_str for x in ['item', 'card', 'product', 'sku', 'cell', 'grid']):
                                    weight = 5
                                clean_cls_list = [c for c in classes if not re.search(r'\d', c) and len(c) > 3]
                                for c in clean_cls_list:
                                    parent_class_counter[f".{c}"] += weight

                potential_selectors = parent_class_counter.most_common(5)
                best_selector = potential_selectors[0][0] if potential_selectors else ".sku-item"
                print(f"[TRACE] Depth {current_depth} Winner Selector: {best_selector}")

                # --- PHASE 3: SEMANTIC EXTRACTION ---
                items = await page.query_selector_all(best_selector)
                for item in items:
                    if len(raw_data) >= 15: break
                    
                    raw_inner_text = await item.inner_text()
                    inner_text_lines = [t.strip() for t in raw_inner_text.split('\n') if t.strip()]
                    if not inner_text_lines: continue

                    potential_titles = inner_text_lines[:6]
                    title = max(potential_titles, key=len) if potential_titles else "N/A"
                    
                    price = "N/A"
                    for line in inner_text_lines:
                        match = price_rex.search(line)
                        if match:
                            price = match.group()
                            break
                    
                    savings = "N/A"
                    for line in inner_text_lines:
                        if "Save" in line:
                            savings = line
                            break

                    if len(title) > 20 and not noise_rex.search(title) and title.lower() not in seen:
                        active_fields = fields if fields else ["Product Name", "Price", "Deals"]
                        active_label = label_context if label_context else ""

                        entry = {}
                        if active_label: entry["context_source"] = active_label

                        if len(active_fields) >= 1: entry[active_fields[0]] = title
                        if len(active_fields) >= 2: entry[active_fields[1]] = price
                        if len(active_fields) >= 3: entry[active_fields[2]] = savings
                        
                        for extra in active_fields[3:]:
                            entry[extra] = "N/A"

                        raw_data.append(entry)
                        seen.add(title.lower())
                
                current_depth += 1

                # --- PHASE 4: NAVIGATION ---
                if current_depth < max_depth and len(raw_data) < 10:
                    next_btn = await page.query_selector('a[rel="next"], button:has-text("Next"), .pagination__next')
                    if next_btn:
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await next_btn.evaluate("el => el.click()")
                    else: break
                else: break

            await browser.close()

        if not raw_data:
            return {"mode": "error", "data": f"No products extracted from {best_selector}. Selector might be too broad."}

        # Build Markdown Table
        dynamic_keys = []
        if raw_data:
            for item in raw_data:
                for k in item.keys():
                    if k not in dynamic_keys and k not in ["context_source"]:
                        dynamic_keys.append(k)
        
        headers = dynamic_keys if dynamic_keys else ["Product Name", "Price", "Deals"]
        table = ["| " + " | ".join([h.replace('_', ' ').title() for h in headers]) + " |", 
                 "| " + " | ".join([":---"] * len(headers)) + " |"]
        for r in raw_data:
            table.append("| " + " | ".join([str(r.get(h, "N/A")).replace("\n", " ") for h in headers]) + " |")
        
        result.update({"data": "\n".join(table), "status": "success"})
        return result

    except Exception as e:
        return {"mode": "error", "data": f"Critical Error: {str(e)}\n{traceback.format_exc()}"}            
                         


def universal_interpreter(recipe, scraper_fn):
    """
    Unified execution environment for Navi. 
    Combines search, extraction, calculation, visualization, and table generation.
    """
    import re
    import json
    import io
    import base64
    import pandas as pd
    import matplotlib.pyplot as plt

    storage = {}
    visuals = []
    logs = []

    # --- INTERNAL UTILITIES ---
    def clean_key(text):
        if not text: return "unlabeled"
        return re.sub(r'\s+', '_', str(text).strip().strip(':*'))

    def to_float(val):
        if val is None: return 0.0
        pattern = r"\$?([\d,.]+)\s*([BMK])?"
        match = re.search(pattern, str(val), re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1).replace(',', ''))
                suffix = match.group(2)
                if suffix:
                    mult = {'B': 1e9, 'M': 1e6, 'K': 1e3}
                    num *= mult.get(suffix.upper(), 1)
                return num
            except: return 0.0
        return 0.0

    def parse_markdown_content(markdown_text, fields, label_context=""):
        results = []
        seen_titles = set()
        
        # 1. Normalize whitelist for comparison
        field_whitelist = [f.lower() for f in fields] if fields else []
        
        actual_content = ""
        if isinstance(markdown_text, dict):
            actual_content = markdown_text.get('data', '')
        elif isinstance(markdown_text, str):
            actual_content = markdown_text
            if actual_content.strip().startswith('{'):
                try: 
                    decoded = json.loads(actual_content)
                    actual_content = decoded.get('data', decoded.get('result', decoded.get('rows', '')))
                except: pass

        if not actual_content or "|" not in str(actual_content):
            return results

        try:
            lines = [l.strip() for l in str(actual_content).split('\n') if l.count('|') >= 2]
            header_idx = -1
            for i in range(len(lines) - 1):
                if re.search(r'\|?\s*:?---', lines[i+1]):
                    header_idx = i
                    break
            if header_idx == -1: return results
            
            headers = [clean_key(h).lower() for h in lines[header_idx].split('|') if h.strip()]
            
            for row_line in lines[header_idx + 2:]:
                cols = [c.strip() for c in row_line.strip('|').split('|')]
                if not cols or len(cols) < 2: continue
                
                extracted = {}
                
                # 2. Inject label_context to track data provenance
                if label_context:
                    extracted['context_source'] = label_context

                # 3. Cell-level extraction (for bolded key-value pairs)
                for col_val in cols:
                    sub_matches = re.findall(r"\*\*(.*?)\*\*[:\s]+(.*?)(?=<br>| \*\*|$)", col_val)
                    for s_key, s_val in sub_matches:
                        key_name = clean_key(s_key).lower()
                        # Apply whitelist filtering
                        if not field_whitelist or key_name in field_whitelist:
                            extracted[key_name] = s_val.strip()
                
                # 4. Table column mapping
                for i, k in enumerate(headers):
                    key_name = k.lower()
                    if i < len(cols) and key_name not in extracted:
                        # NEW: If fields is provided, strict enforcement
                        if field_whitelist:
                            if key_name in field_whitelist:
                                extracted[key_name] = cols[i]
                        else:
                            # If no fields provided, keep original behavior (capture all)
                            extracted[key_name] = cols[i]

                # Filtering and Deduplication
                p_title = str(extracted.get("title", extracted.get("name", extracted.get("product_name", "")))).lower()
                if any(meta in p_title for meta in ["portal:", "help:", "special:", "talk:"]):
                    continue

                unique_id = p_title if p_title else str(extracted)
                if unique_id not in seen_titles:
                    # Only append if we actually extracted relevant fields
                    if len(extracted) > (1 if label_context else 0):
                        results.append(extracted)
                        seen_titles.add(unique_id)
                    
        except Exception as e:
            # Safely check for logs list before appending
            if 'logs' in globals() or 'logs' in locals():
                logs.append(f"⚠️ Scavenge Error: {str(e)[:50]}")
        return results

    # --- MAIN RECIPE LOOP ---
    for step in recipe:
        if not isinstance(step, dict): continue
        action = step.get("action")
        params = step.get("params", {})

        if action in ["gather", "scrape_direct"]:
            label = clean_key(params.get('label', 'data'))
            query = params.get('url', params.get('search_query', ''))
            
            if not query:
                logs.append(f"❌ No URL/Query for {label}")
                continue
                
            url = query if query.startswith('http') else None
            # Placeholder for Search Engine if URL is missing
            if not url:
                logs.append(f"🔍 Search required for: {query}")
                continue
            

            raw_payload = run_sync_scraper(
                scraper_fn,
                url.split('#')[0], 
                params.get('task_query', 'Extract details'),
                params.get('max_depth', 0),
                fields=params.get('fields', []),
                label_context=params.get('label', '')
            )
            new_items = parse_markdown_content(raw_payload, params.get('fields', []), label)
            
            if label not in storage: storage[label] = []
            
            if new_items:
                storage[label].extend(new_items)
                logs.append(f"✅ {label}: Found {len(new_items)} items.")
            else:
                storage[label].append({'raw_text': str(raw_payload)[:300]})
                logs.append(f"⚠️ {label}: No structured items.")

        elif action == "calculate":
            # FIXED: Defensive get() for KeyError prevention
            label = clean_key(params.get('label', 'calc_result'))
            formula = params.get('formula', '')
            if not formula:
                logs.append(f"⚠️ Missing formula for {label}")
                continue

            try:
                flat_vars = {}
                for s_label, items in storage.items():
                    if isinstance(items, list):
                        for idx, item in enumerate(items):
                            for field, val in item.items():
                                flat_vars[f"{s_label}_{field}_{idx}"] = to_float(val)
                                if idx == 0: flat_vars[f"{s_label}_{field}"] = to_float(val)
                
                # Replace variable placeholders
                final_formula = formula
                for var_name, var_val in flat_vars.items():
                    final_formula = final_formula.replace(f"{{{{{var_name}}}}}", str(var_val))
                
                safe_formula = re.sub(r'[^0-9./*+()-]', '', final_formula)
                calc_result = eval(safe_formula, {"__builtins__": None}, {}) 
                
                if "calculations" not in storage: storage["calculations"] = []
                storage["calculations"].append({label: calc_result})
                logs.append(f"🧮 {label} = {calc_result}")
            except Exception as e: 
                logs.append(f"⚠️ Calc Error: {e}")

        elif action == "visualize":
            try:
                metric = params.get('metric')
                if not metric: continue
                
                plt.figure(figsize=(8, 5))
                plot_labels, plot_values = [], []

                for l, items in storage.items():
                    if l == 'calculations' or not isinstance(items, list): continue
                    for i in items:
                        if metric in i:
                            plot_labels.append(str(i.get('product_name', i.get('title', i.get('name', l))))[:15])
                            plot_values.append(to_float(i[metric]))

                if plot_values:
                    chart_type = params.get('type', 'bar')
                    if chart_type == 'bar': plt.bar(plot_labels, plot_values, color='skyblue')
                    else: plt.plot(plot_labels, plot_values, marker='o', color='orange')
                    plt.title(params.get('title', f"Trend: {metric}"))
                    plt.xticks(rotation=45)
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    visuals.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
                    plt.close()
                    logs.append(f"📊 {chart_type.capitalize()} chart ready.")
            except Exception as e:
                logs.append(f"📊 Viz Error: {e}")

        elif action == "table":
            try:
                all_rows = []
                for k, v in storage.items():
                    if isinstance(v, list) and k != 'calculations':
                        for item in v:
                            row = {"Source": k}
                            row.update(item)
                            all_rows.append(row)
                if all_rows:
                    storage['formatted_table'] = pd.DataFrame(all_rows).to_markdown()
                    logs.append("📋 Generated summary table.")
            except: pass

    return {
        "final_answer": json.dumps(storage), 
        "image_payload": visuals,
        "execution_logs": "\n".join(logs)
    }
    

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

import os
import sys
import subprocess
import platform

def ensure_packages(package_list):
    """
    Installs missing packages safely and handles Playwright's binary requirements.
    Optimized for Streamlit Cloud (/mount/src/...) and Local environments.
    """
    failed_packages = []
    
    # 1. Determine the persistent browser path
    # On Streamlit, we want a fixed path: /mount/src/bi-agent-v2/pw-browsers
    # On Local, we use a 'pw-browsers' folder in your current directory
    if os.path.exists("/mount/src/bi-agent-v2"):
        base_pw_path = "/mount/src/bi-agent-v2/pw-browsers"
    else:
        base_pw_path = os.path.join(os.getcwd(), "pw-browsers")

    for package in package_list:
        # Normalize the name for the __import__ check
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
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                
                # --- THE PLAYWRIGHT SPECIAL CASE ---
                if package.lower() == "playwright":
                    print(f"🚀 Initializing Playwright in: {base_pw_path}")
                    os.makedirs(base_pw_path, exist_ok=True)
                    
                    # Update the current process environment immediately
                    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = base_pw_path
    
                    env = os.environ.copy()
                    env["PLAYWRIGHT_BROWSERS_PATH"] = base_pw_path
    
                    # CRITICAL: Install chromium AND chromium-headless-shell
                    # This ensures the shell binary exists for our scraper's specific pathing
                    print("⬇️ Downloading Chromium & Headless Shell...")
                    subprocess.check_call(
                        [sys.executable, "-m", "playwright", "install", "chromium", "chromium-headless-shell"], 
                        env=env
                    )
                    print("✅ Playwright binaries prepared.")

            except subprocess.CalledProcessError as e:
                print(f"❌ System: Failed to install {package} (Exit code: {e.returncode})")
                failed_packages.append(package)
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
    UNIVERSAL PLANNER NODE - V2.2 (Strict Escalation Edition)
    Manages the lifecycle of a task through automated repair, 
    external research, and philosophical meditation.
    """
    final_ans_raw = state.get("final_answer")
    last_error = state.get("last_error")
    retry_count = state.get("retry_count", 0)
    plan = state.get("plan", [])
    task = state.get("task")
    plan_str = str(plan)
    meditation_notes = state.get("meditation_notes")
    
    # ---------------------------------------------------------
    # 1. ESCALATION & TERMINATION (The "Panic" Logic)
    # ---------------------------------------------------------
    # returning from Meditation recovery
    if meditation_notes:
        print(f"🧘 Planner: Meditation insights received. Routing to Final Recovery Attempt.")
        return {
            "plan": plan + ["### 🛠️ ACTION: CODE"],
            "retry_count": retry_count + 1,
            "last_error": None,
            "meditation_notes": None # Reset to prevent loop
        }

    # If we've already tried everything and reached this node again
    if retry_count >= 5: 
        print("🛑 Planner: All recovery tiers exhausted. Shifting to Conversational Mode.")
        return {
            "plan": plan + ["### 💬 ACTION: CONVERSE"],
            "is_terminal": True # This will be the signal for the Router
        }

    # ---------------------------------------------------------
    # 2. AUDIT & RECURSION GATE (The "Escalation" Logic)
    # ---------------------------------------------------------
    if final_ans_raw:
        print(f"🧐 Planner: Auditing Cycle {retry_count}...")
        print(final_ans_raw)
        # Check if the data is actually empty or just a "No Results" message
        is_actually_empty = any(x in str(final_ans_raw).lower() for x in ["no matching items", "no data", "not found", "[]", "{}"])
        
        audit_prompt = f"""
        ### OBJECTIVE: {task}
        ### DATA PAYLOAD: {str(final_ans_raw)[:3000]}
        
        ### CRITICAL INSTRUCTION:
        You are a Semantic Data Auditor. Your job is NOT to grade the formatting, but to verify if the 'Gold' (the actual requested information) is present in the text.
        
        **SEMANTIC COMPLETENESS**: If the requested information is "there or mostly there," even if it is mixed with noise or incomplete fragments, you MUST accept it. 
        **FORMAT AGNOSTIC**: Do not reject the data because it is in a list, a messy JSON, or a broken table. Content is king.

        ### DECISION LOGIC:
        -If the information is found in the content of the {final_ans_raw} you MUST respond COMPLETE.  It is up to you to find this information even if it's buried in the text.  READ through ALL of the text in {final_ans_raw} before your response and if data is present you MUST respond COMPLETE
        - ONLY respond with CONTINUE if the data is genuinely empty, 100% unrelated (e.g., only captured the footer), or a "403 Forbidden" error. If there is data you MUST respond with COMPLETE

        Respond with your brief analysis, then end with exactly one word: COMPLETE or CONTINUE.
        """
        
        audit_raw = llm_fast.invoke(audit_prompt).content.strip()
        audit_decision = "CONTINUE" if "CONTINUE" in audit_raw.upper() else "COMPLETE"
        
        # --- THE ESCALATION TRIGGER ---
        if audit_decision == "CONTINUE":
            # Increment the failure count even if the code 'ran' successfully
            failures = state.get("failure_count", 0) + 1
            
            # ESCALATION THRESHOLD: 
            # If we have tried 3 times (Cycle 0, 1, 2) and still have nothing...
            if failures >= 3:
                print("🚨 ESCALATION: Scraper logic is circular. Pivoting to RESEARCH mode.")
                return {
                    "plan": plan + ["### 🔍 ACTION: RESEARCH (Automated scraping exhausted; initiating manual intelligence gathering)"],
                    "is_continue": True,
                    "auditor_notes": "Automated scraping failed to find data after 3 attempts. Need manual search for better URLs.",
                    "failure_count": failures,
                    "retry_count": retry_count + 1,
                    
                }
            
            # Standard Re-try logic
            return {
                "plan": plan + [f"### 🔄 ACTION: CODE (Cycle {failures})"],
                "is_continue": True,
                "auditor_notes": audit_raw.split('|')[0],
                "failure_count": failures,
                "retry_count": retry_count + 1
            }

        # --- FINAL SYNTHESIS ---
        print("✨ Planner: Logic satisfied. Synthesizing final response.")
        # (Rest of your synthesis logic remains the same...)
        summary_response = llm_fast.invoke(f"""
        MISSION CRITICAL DATA RECONSTRUCTION:
        Target Task: '{task}'
        
        RAW SCRAPER OUTPUT:
        {final_ans_raw}

        INSTRUCTION: 
        Extract and summarize the data above. If you see the core answers, format them into a clean list or table and finalize the response now. Never output raw data directly, always summarize it first.
    """).content.strip()

        return {
            "final_answer": summary_response, 
            "last_error": None,               
            "plan": plan + ["### 🏁 ACTION: EXIT"],
            "is_continue": False,
            "auditor_notes": None
        }

    # ---------------------------------------------------------
    # 3. DYNAMIC ERROR RECOVERY LADDER
    # ---------------------------------------------------------
    if last_error:
        
        if retry_count < 2:
            return {
                "plan": plan + ["### 🛠️ ACTION: CODE"],
                "last_error": last_error,
                "retry_count": retry_count + 1
            }
        
        
        elif "ACTION: RESEARCH" not in plan_str:
            print("🔍 Planner: Code repair failed. FORCING RESEARCH escalation.")
            return {
                "plan": plan + ["### 🔍 ACTION: RESEARCH"],
                "retry_count": 0 # Reset count so we have fresh tries after research
            }
        
        elif "ACTION: MEDITATE" not in plan_str:
            print("🧘 Planner: Research complete but issues persist. Escalating to MEDITATION.")
            return {
                "plan": plan + ["### 🧘 ACTION: MEDITATE"],
                "retry_count": retry_count + 1
            }
        
        
        else:
            print("🛑 Planner: All recovery tiers exhausted. Moving to Conversational.")
            return {
                "plan": plan + ["### 🏁 ACTION: CONVERSE"],
                "is_terminal": True
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

    # Initial Entry
    if not plan:
        print("🚀 Planner: Initiating new Task Sequence.")
        return {
            "plan": ["### 🛠️ ACTION: CODE"], 
            "retry_count": 0
        }
    
    return {}


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
    fail_count = state.get("failure_count", 0) 
    auditor_notes = state.get("auditor_notes", "")
    is_continue = state.get("is_continue", False)
    
    past_strategies = state.get("past_strategies", [])
    failed_code = state.get("generated_tool_code")
    res_fails = state.get("consecutive_research_failures", 0) + 1
    logs = state.get("execution_logs", "No logs found.")
    
    # 1. Error Identification (Combining Technical & Logical Failures)
    if not raw_error and auditor_notes:
        last_error = f"Logical Failure: {auditor_notes}"
    elif raw_error:
        last_error = str(raw_error)
    else:
        last_error = "Unknown failure: The process stalled without an error message."

    if res_fails >= 3:
        print("🛑 CRITICAL: Maximum research attempts reached. System saturation.")
        return {
            "consecutive_research_failures": res_fails,
            "plan": state.get("plan", []) + ["### 🏁 ACTION: EXIT (Research Exhausted)"]
        }
    
    print(f"\n🔍 RESEARCHER ACTIVATED (Strategy Pivot {res_fails}/3)")

    # 2. Meta-Analysis Prompt
    # This instructs the model to use NaviEngine's search if the URL itself is the problem.
    reasoning_prompt = f"""
    ### 🧠 META-ANALYSIS TASK
    Objective: {task}
    Current Cycle: {state.get('retry_count', 0)}
    
    ### 🚫 FAILURE ANALYSIS
    - **Logic Tried**: {failed_code if failed_code else "None"}
    - **Auditor Observation**: {auditor_notes if auditor_notes else "None"}
    - **Technical Error**: {last_error}
    
    ### 🛠️ STRATEGY PIVOT
    You must diagnose why the NetNavi is failing. 
    1. If the URL provided was wrong or blocked, suggest a DuckDuckGo search query to find a better deep-link.
    2. If the CSS selectors were wrong, suggest a text-based search (e.g., finding the word 'Next' instead of '.pagination-btn').
    3. If the Scraper hit a loop, provide a 'Jailbreak' instruction for the Skill Creator.

    ### 📋 OUTPUT FORMAT (STRICT JSON)
    {{
        "diagnosis": "Detailed summary of the failure and the specific fix.",
        "solution_logic": "Step-by-step logic for the next attempt.",
        "search_suggestion": "Optional DDG search query if the entry URL is bad.",
        "recommended_fields": ["updated", "field", "list"]
    }}
    """
    
    try:
        # We always want a 'smarter' brain for research to prevent the 8B model from repeating loops
        print("🌙 Researcher: Consulting 70B for Strategic Pivot...")
        response = llm_pro.invoke(reasoning_prompt)
        raw_output = response.content
    except Exception as e:
        print(f"🔄 Researcher Fallback: {e}")
        response = llm_fast.invoke(reasoning_prompt)
        raw_output = response.content

    # 3. JSON Cleaning & NaviEngine Integration
    clean_json = re.sub(r'^```json\s*|```$', '', raw_output, flags=re.MULTILINE | re.IGNORECASE).strip()
    
    try:
        data = json.loads(clean_json)
        diagnosis = data.get("diagnosis", "No diagnosis provided.")
        pivot_strategy = data.get("solution_logic", "")
        
        # If the researcher suggests a new search, we could trigger NaviEngine.run_search here
        # For now, we package it into the research_notes for the Skill Creator
        research_notes = f"DIAGNOSIS: {diagnosis}\nSTRATEGY: {pivot_strategy}"
        if data.get("search_suggestion"):
            research_notes += f"\nRECOMMENDED SEARCH: {data['search_suggestion']}"

    except Exception as e:
        diagnosis = f"Structural Logic Analysis: {clean_json[:200]}"
        research_notes = clean_json

    # 4. Return updated state
    # We clear the auditor_notes and is_continue so the loop can start fresh with the new strategy
    return {
        "research_notes": research_notes, 
        "last_error": None, 
        "auditor_notes": None,
        "is_continue": False, 
        "consecutive_research_failures": res_fails,
        "plan": state.get("plan", []) + [f"### 🔍 STRATEGY PIVOT: {diagnosis}..."],
        "retry_count": state.get("retry_count", 0) # Keep retry count persistent
    }
            

# --- Node 2: Skill Creator (The Self-Learning Node) ---

import json
import re
import ast

def skill_creator_node(state: NaviState):
    task = state.get('task', "")
    plan = state.get('plan', [])
    last_error = state.get('last_error', "None")
    research_notes = state.get("research_notes", "")
    history = state.get("history", [])
    
    # --- 1. RECURSION & AUDITOR CONTEXT ---
    is_continue = state.get("is_continue", False)
    auditor_notes = state.get("auditor_notes", "")
    retry_count = state.get("retry_count", 0)

    # If is_continue is True, inject the Auditor's "Diagnostic Note" as a strict constraint
    recursion_injection = ""
    if is_continue and auditor_notes:
        recursion_injection = f"""
### 🔄 RECURSION ALERT: PREVIOUS ATTEMPT INCOMPLETE
**Auditor Diagnostic**: "{auditor_notes}"
**Last Error**: {last_error}
**Instruction**: Your previous logic recipe did not satisfy the goal. You MUST pivot your strategy. 
- Change your 'search_query' to a more specific sub-page or category index.
- Refine your 'task_query' to be more descriptive of the elements on the page.
- Ensure the 'fields' requested are present in the target data.
"""

    # --- 2. LOGIC ARCHITECT PROMPT (NaviEngine Integrated) ---
    prompt = f"""
### TASK DEFINITION
Generate a JSON execution recipe to fulfill: {task}
{recursion_injection}

### ALLOWED ACTIONS
1. "gather": URL/Search to Detail. Params: 'label', 'search_query', 'task_query', 'fields', 'target_count'.
2. "scrape_direct": Specific URL. Params: 'label', 'url', 'task_query', 'fields'.
3. "calculate": Math. Use `{{{{Label_Field}}}}`. Params: 'label', 'formula'.
4. "table": Markdown summary. Params: 'title'.
5. "visualize": Bar/Line charts. Params: 'title', 'type', 'metric'.

### STRICT EXECUTION RULES
- **NO ELLIPSIS**: Never use `...` or placeholders in values. Every JSON key must have a complete string, number, or list value.
- **NO CONTEXTUAL SHORTHAND**: Do not assume the interpreter knows what to fill in. Provide full formulas and full field lists.
- **OUTPUT ONLY VALID JSON**: You MUST begin your output with '[' and END with ']'.  In between should be JSON ONLY, NEVER include ```json in your response.
- **ERROR RECOVERY**: {research_notes if research_notes else "Fix any previous serialization or syntax errors."}

### MANDATORY SCHEMA
[
  {{
    "action": "gather",
    "params": {{
      "label": "articles",
      "search_query": "https://www.theverge.com/archives",
      "task_query": "Get latest 5 tech headlines",
      "fields": ["title", "summary", "topic"]
    }}
  }},
  {{
    "action": "calculate",
    "params": {{ 
        "label": "why_it_matters", 
        "formula": "'Synthesis of top tech trends'" 
    }}
  }}
]

### JSON OUTPUT:
"""
    
    # --- 3. INTELLIGENCE WATERFALL ---
    try:
        # Use Pro model for recursion/complex tasks, Fallback to Flash for standard runs
        if is_continue or retry_count > 0:
            print(f"🌙 Skill Creator: Escalating to 70B (Recursion Cycle {retry_count})...")
            response = llm_pro.invoke(prompt)
        else:
            print("🌙 Skill Creator: Attempting Logic Architecture...")
            response = llm_pro.invoke(prompt)
        raw_output = response.content
    except Exception as e:
        print(f"🔄 Fallback to 8B: {e}")
        response = llm_fast.invoke(prompt)
        raw_output = response.content
        
    # --- 4. JSON CLEANING & STATE COMMIT ---
    try:
        # Extract JSON array from potential fluff
        match = re.search(r"(\[.*\])", raw_output, re.DOTALL)
        
        if match:
            clean_json_str = match.group(1).strip()
            recipe_data = json.loads(clean_json_str)
            generated_tool_code = json.dumps(recipe_data, indent=2)
        else:
            generated_tool_code = raw_output.strip()

        print(f"DEBUG A: Skill Creator sending: {len(generated_tool_code)} chars")
        
        # We clear is_continue here; the Auditor will re-set it if the new recipe also fails
        return {
            "generated_tool_code": generated_tool_code,
            "plan": plan + [f"### 🧠 Logic Recipe Refined (Cycle {retry_count})"],
            "is_continue": False,
            "last_error": None
        } 
        
    except Exception as e:
        print(f"❌ Skill Creator Parsing Error: {e}")
        return {
            "generated_tool_code": raw_output, 
            "last_error": f"Logic parsing failed: {str(e)}",
            "is_continue": False
        }
                            


# --- Node 3: Executor ---
import json
import re
import os
import subprocess
import tempfile
import textwrap
import sys

def executor_node(state: NaviState):
    print("\n🚀 [SUBPROCESS] Starting Lite Execution...")
    
    # This ensures even if the key is None, it defaults to an empty string BEFORE stripping
    code = (state.get('generated_tool_code') or "").strip()
    print(f"DEBUG B: Executor received: {type(code)} | Value: {code[:100]}...") # Truncated print for readability
    
    packages = state.get('packages', []) 
    retry_count = state.get('retry_count', 0)
    current_plan = state.get('plan', [])
    task = state.get("task")
    
    if not code:
        return {
            "last_error": "### ❌ Execution Failed\nNo code or recipe found.",
            "retry_count": retry_count + 1,
            "plan": current_plan + ["### ❌ No Logic to Execute"],
            "generated_tool_code": "" 
        }

    # --- PATH A: UNIVERSAL RECIPE INTERPRETER (JSON) ---
    if code.startswith("[") and code.endswith("]"):
        try:
            print("🧩 Detected JSON Recipe. Routing to Universal Interpreter...")
            recipe = json.loads(code)
            import asyncio
            # Execute
            results = run_sync_scraper(universal_interpreter, recipe, universal_scraper)
            
            # ✅ CORRECT MAPPING:
            # The interpreter now returns 'final_answer', 'image_payload', and 'execution_logs'
            final_text_result = results.get('final_answer', '{}')
            image_payloads = results.get('image_payload', [])
            logs = results.get('execution_logs', "")

            return {
                "final_answer": final_text_result,
                "image_payload": image_payloads, 
                "last_error": None,
                "executor_logs": logs,
                "retry_count": retry_count,
                "generated_tool_code": code,  
                "plan": current_plan + ["### ✅ Recipe Execution Success"]
            }
        
        except Exception as e:
            import traceback
            # This will print the EXACT line number and error type in your terminal
            print(f"❌ INTERPRETER CRASHED: {traceback.format_exc()}") 
            
            return {
                "last_error": f"### ❌ Recipe Interpreter Error\n{str(e)}",
                "executor_logs": traceback.format_exc(),
                "retry_count": retry_count + 1,
                "generated_tool_code": code, 
                "plan": current_plan + [f"### ❌ Recipe Failed: {str(e)}"]
            }

    # --- PATH B: LEGACY PYTHON SUBPROCESS (For raw code generation) ---
    import inspect
    import asyncio

    try:
        # This still gets the source, but it will now start with "async def..."
        scraper_code = inspect.getsource(universal_scraper)
        
        # We append a 'wrapper' to the scraper_code so that the 
        # generated script knows how to run the async function
        scraper_code += """
    def sync_wrapper(*args, **kwargs):
        import asyncio
        return asyncio.run(universal_scraper(*args, **kwargs))
    """
    except Exception as e:
        print(f"⚠️ Warning: Could not find universal_scraper source: {e}")
        scraper_code = "def universal_scraper(url): return 'Error: Scraper source missing.'"

    raw_script = f"""
import sys
import io
import json
import re
import os
import base64

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# --- INJECTED UTILITIES ---
{scraper_code}

# --- NAVI GENERATED CODE ---
{code}

if __name__ == "__main__":
    try:
        if 'execute_tool' not in globals():
            sys.stdout.write("EXECUTION_ERROR: Function 'execute_tool' not defined in generated code.\\n")
        else:
            result = execute_tool()
            output = str(result) if result is not None else "NAVI_EMPTY_RESULT"
            sys.stdout.write("\\n---NAVI_RESULT_START---\\n")
            sys.stdout.write(output)
            sys.stdout.write("\\n---NAVI_RESULT_END---\\n")
    except Exception as e:
        sys.stdout.write(f"EXECUTION_ERROR: {{e}}\\n")
"""

    full_script = textwrap.dedent(raw_script).strip()

    # Dynamic Package Installation Logic
    internal_tools = ["universal_scraper", "ddgs_search", "NaviState"] 
    filtered_packages = [p for p in packages if p not in internal_tools]
    if filtered_packages:
        for pkg in filtered_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            except: pass

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
            f.write(full_script)
            temp_path = f.name

        env_vars = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        process = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=120, env=env_vars)
        
        output_cleaned = process.stdout + "\n" + process.stderr
        output_cleaned = re.sub(r"\[notice\].*?|WARNING: Running pip.*?", "", output_cleaned).strip()

        print(f"\n{'='*20} SUBPROCESS LOGS {'='*20}\n{output_cleaned}\n{'='*48}")

        match = re.search(r"---NAVI_RESULT_START---\s*(.*?)\s*---NAVI_RESULT_END---", output_cleaned, re.DOTALL)
        
        if match:
            extracted_data = match.group(1).strip()
            
            image_payloads = []
            b64_pattern = r"(iVBORw0KGgoAAAANSUhEUg[A-Za-z0-9\+/=\s\n]+)"
            all_figs = re.findall(b64_pattern, extracted_data)
            
            clean_for_llm = extracted_data
            if len(clean_for_llm) > 5000:
                clean_for_llm = f"{clean_for_llm[:2500]}\n\n... [TRUNCATED] ...\n\n{clean_for_llm[-2500:]}"
            for idx, fig_raw in enumerate(all_figs):
                image_data_clean = re.sub(r"[^A-Za-z0-9\+/=]", "", fig_raw)
                image_payloads.append(image_data_clean)
                clean_for_llm = clean_for_llm.replace(fig_raw, f"\n\n[IMAGE_DATA_HIDDEN_{idx}]\n\n")
            
            return {
                "final_answer": clean_for_llm,
                "image_payload": image_payloads,
                "last_error": None,
                "executor_logs": output_cleaned,
                "retry_count": retry_count,
                "generated_tool_code": code, 
                "plan": current_plan + ["### ✅ Legacy Code Success"]
            }

        return {
            "last_error": f"### ❌ Execution Failed\n{output_cleaned[-500:]}",
            "executor_logs": output_cleaned,
            "retry_count": retry_count + 1,
            "generated_tool_code": code, 
            "plan": current_plan + ["### ❌ Code Failed"]
        }

    finally:
        if temp_path and os.path.exists(temp_path):
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
    last_step = str(plan[-1]).upper() if plan else ""
    
    if "CONVERSE" in last_step:
        return "conversational"
    
    if any(k in last_step for k in ["EXIT", "TERMINATED"]):
        return "memory_save"
    
    if "RESEARCH" in last_step:
        return "research"
    
    if "MEDITATE" in last_step:
        return "meditator"
    
    return "skill_creation"


def route_after_research(state: NaviState):
    plan = state.get("plan", [])
    last_step = str(plan[-1]).upper()
    research_fails = state.get("consecutive_research_failures", 0)
    
    # If research fails twice, move to Meditator
    if "EXIT" in last_step or research_fails >= 2:
        print("🧘 Route: Research hit a wall. Triggering Meditation.")
        return "meditator"

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
    lambda state: "planner" if not state.get("is_terminal") else "conversational",
    {
        "planner": "planner",
        "conversational": "conversational"
    }
)

# --- 3. UPDATED EXIT LOGIC: Save before END ---
# Replace END with "memory_save" in all final steps


workflow.add_conditional_edges(
    "planner", 
    route_after_plan, 
    {
        "skill_creation": "skill_creation", 
        "research": "research", 
        "meditator": "meditator",
        "conversational": "conversational", 
        "memory_save": "memory_save"
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
