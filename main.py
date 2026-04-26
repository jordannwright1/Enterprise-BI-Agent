import streamlit as st
import os
from dotenv import load_dotenv
import uuid
from core.graph import navi_app
from langchain_core.messages import HumanMessage
import re
from core.database import init_db
import sqlite3
import base64
from PIL import Image
import io
import sys

# --- 1. CONFIGURATION & ENVIRONMENT ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Force Playwright to use a local folder within the Streamlit mount
local_bins = os.path.join(ROOT_DIR, ".playwright_bins")
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = local_bins
os.makedirs(local_bins, exist_ok=True)

load_dotenv()
init_db()

def display_navi_chart(b64_string):
    try:
        clean_b64 = b64_string.strip()
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",")[-1]
        img_data = base64.b64decode(clean_b64)
        img = Image.open(io.BytesIO(img_data))
        st.image(img, caption="Navi's Generated Analysis", use_container_width=True)
    except Exception as e:
        st.error(f"Could not render chart: {e}")

def inspect_skills():
    conn = sqlite3.connect("tools/navi_skills.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS skills (id INTEGER PRIMARY KEY, keyword TEXT, created_at TEXT)")
    cursor.execute("SELECT id, keyword, created_at FROM skills")
    rows = cursor.fetchall()
    conn.close()
    if rows:
        for row in rows:
            print(f"ID: {row[0]} | Skill: {row[1]} | Learned On: {row[2]}")

inspect_skills()

def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        if st.button("🗑️ Reset Agent Memory", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            new_id = str(uuid.uuid4())
            st.session_state.thread_id = new_id
            st.session_state.messages = []
            st.success(f"Brain Wipe Successful. New Thread: {new_id[:8]}")
            st.rerun()
        st.markdown("---")
        st.subheader("💡 Example Prompts")
        example_prompts = [
             """Generate a Cross-Platform Tech Intelligence Report.

            Strategic Moves (TechCrunch): Go to https://techcrunch.com/ and scrape the first 3 articles. Extract the Author, summarize the entire article in a paragraph summary, and create a one to two sentence summary of the Strategic Impact from the full article pages.

            Product & Culture (The Verge): Go to https://www.theverge.com/ and scrape the first 2 featured stories. Create a paragraph summary for each article, and the list Primary Subject from the full article pages.

            Synthesis Requirement: Create a single 'Executive Briefing' that contrasts the Business/Financial focus of TechCrunch with the Consumer/Product focus of The Verge. Provide a 'Recruiter's Perspective' on which tech sectors (e.g., AI, Fintech, Hardware) are showing the most momentum today based on these 5 sources.""","""Run a 12-month Monte Carlo simulation (1,000 trials) for a B2B SaaS startup.
            Parameters: > 1. Initial Burn: $150k starting capital.
            2. Revenue: $100 Monthly ARPU (Average Revenue Per User).
            3. Acquisition: Normal distribution (Mean=40, StdDev=10) new customers/mo.
            4. Retention: Model a 5% monthly churn rate (compounding).
            5. Costs: Fixed $8k/mo + variable $5/user/mo for infrastructure.

            Task: Visualize the Cash Runway over 12 months. Plot the 5th, 50th, and 95th percentiles of net cash flow. Based on the 'Worst Case' (5th percentile), tell me exactly which month we run out of money and recommend an ARPU or Churn adjustment to survive 18 months.""", 
        ]
        for ex in example_prompts:
            st.code(ex, language=None)

st.set_page_config(page_title="Navi: Autonomous Agent", page_icon="🌐", layout="wide")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# --- 2. SESSION STATE MANAGEMENT ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"navi_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Navi: Self-Learning Multi-Purpose Agent")
st.markdown("Navi is an advanced AI agent that thinks, researches, generates data visualizations, and executes code in real-time. This agent maintains a skill library repository and can both load and create new skills.  You can speak to the agent conversationally as well and it will remember your previous messages. It can also scrape webpages from multiple sources and generate syntheses and reports. Use the sidebar to copy one of the test prompts, or create your own prompt.")
st.markdown("---")
render_sidebar()


# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            for img_str in msg["images"]:
                display_navi_chart(img_str)

# --- 3. EXECUTION LOOP ---
if prompt := st.chat_input("Assign a task to Navi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 50}
    
    # Updated initial_input with Memory Keys and Scraper
    initial_input = {
        "user_input": prompt,
        "task": prompt,
        "inventory": ["ddgs_search", "universal_scraper", "read_file", "write_file"],
        "retry_count": 0,
        "loop_count": 0,
        "consecutive_failures": 0,
        "consecutive_research_failures": 0,
        "meditation_triggered": False,
        "history": [HumanMessage(content=prompt)],
        "memory_context": "",       # New Key: For retrieved context
        "memories_to_save": [],     # New Key: For storage
        "final_answer": None,
        "last_error": None,
        "generated_tool_code": None,
        "packages": [],
        "image_payload": [],
        "plan": [] 
    }

    with st.chat_message("assistant"):
        final_ans = None
        current_image_payload = []
        
        with st.status("Navi is processing...", expanded=True) as status:
            for event in navi_app.stream(initial_input, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    
                    # DEFENSIVE GUARD: Skip if node returns None or empty
                    if node_output is None:
                        continue

                    if node_name == "memory_recall":
                        if node_output.get("memory_context"):
                            st.write("🧠 *Navi recalled relevant past context...*")

                    if node_name == "meditator":
                        st.warning("🧘 Navi is meditating on a persistent error...")
                        if node_output.get("meditation_notes"):
                            with st.expander("View Root Cause Analysis"):
                                st.write(node_output["meditation_notes"])

                    if "plan" in node_output and node_output["plan"]:
                        plan_text = node_output["plan"][-1]
                        st.write(f"📝 {plan_text}")
                    
                    if "final_answer" in node_output and node_output["final_answer"]:
                        final_ans = node_output["final_answer"]

                    if "image_payload" in node_output and node_output["image_payload"]:
                        for img in node_output["image_payload"]:
                            if img not in current_image_payload:
                                current_image_payload.append(img)
            
            # State Synchronization
            final_state = navi_app.get_state(config).values
            final_ans = final_ans or final_state.get("final_answer")
            current_image_payload = current_image_payload or final_state.get("image_payload", [])

            status.update(label="Process Finished", state="complete", expanded=False)

        # --- FINAL RENDERING LOGIC (PRESERVED EXACTLY) ---
        if final_ans:
            placeholders = re.findall(r"\[IMAGE_DATA_HIDDEN_(\d+)\]", final_ans)
            stored_images = []

            # SCENARIO A: INLINE MULTI-IMAGE INJECTION
            if placeholders and current_image_payload:
                split_pattern = r"(?:Figure:\s*|Plot:\s*)?\(?\[IMAGE_DATA_HIDDEN_\d+\]\)?[:\s]*"
                parts = re.split(split_pattern, final_ans)
                for i, part in enumerate(parts):
                    if part.strip(): st.markdown(part.strip())
                    if i < len(current_image_payload):
                        display_navi_chart(current_image_payload[i])
                        stored_images.append(current_image_payload[i])
                st.session_state.messages.append({"role": "assistant", "content": final_ans, "images": stored_images})

            # SCENARIO B: SINGLE GENERIC PLACEHOLDER
            elif "[IMAGE_DATA_HIDDEN]" in final_ans and current_image_payload:
                display_text = final_ans.replace("[IMAGE_DATA_HIDDEN]", "").replace("Figure:", "").strip()
                st.markdown(display_text)
                img_to_show = current_image_payload[0]
                display_navi_chart(img_to_show)
                st.session_state.messages.append({"role": "assistant", "content": display_text, "images": [img_to_show]})

            # SCENARIO C: NO PLACEHOLDERS BUT IMAGES EXIST
            elif current_image_payload:
                st.markdown(final_ans)
                for img_str in current_image_payload:
                    display_navi_chart(img_str)
                st.session_state.messages.append({"role": "assistant", "content": final_ans, "images": current_image_payload})

            # SCENARIO D: TEXT ONLY
            else:
                st.markdown(final_ans)
                st.session_state.messages.append({"role": "assistant", "content": final_ans})
