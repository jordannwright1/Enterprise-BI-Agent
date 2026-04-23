# Navi: A Self-Evolving Autonomous Research & Business Intelligence Agent

Navi is a "Digital Brain" architecture built on LangGraph, designed to bridge the gap between static LLM responses and complex, multi-step professional services automation. Unlike standard chatbots, Navi possesses a **recursive execution loop** that allows her to research, develop, and persist her own library of executable Python skills.

## 🧠 Architectural Philosophy: The Digital Connectome
Navi is modeled conceptually after biological neural structures, organized into functional "nuclei" (nodes) that communicate via a state-based "synaptic" flow:

* **Prefrontal Cortex (The Planner):** Decomposes complex human intent into actionable multi-step research and execution strategies.
* **Hippocampus (The Skill DB):** A persistent database where verified, successful code blocks are stored as "skills" for future recall, bypassing the need for redundant computation.
* **The Meditator (Self-Healing Loop):** A dedicated error-handling node that triggers when execution fails. It "meditates" on the traceback, identifies environment mismatches (e.g., binary encoding errors or missing headers), and pivots the strategy.
* **Motor Cortex (The Executor):** A secure subprocess environment that dynamically installs dependencies and executes Python code in real-time.

## 🚀 Core Capabilities & Skills

### 📊 Advanced Data Visualization
Navi generates production-grade, labeled visualizations and strategic models, including:
* **Market Profitability Heatmaps:** Multi-dimensional grids visualizing "Currency Risk" vs. "Pricing Power" to determine optimal international entry points.
* **Competitive Analysis Radar Charts:** Multi-variable comparisons of market entities (e.g., assessing Coffee shop chains by price, quality, accessibility, and brand sentiment).
* **Monte Carlo Risk Simulations:** Stochastic modeling of business outcomes, using thousands of simulated trials to visualize the probability distribution of ROI under volatile market conditions.

### 🔬 Strategic Modeling & Simulation
* **Monte Carlo Risk Analysis:** Performs probabilistic forecasting by running thousands of simulated trials on volatile business variables (e.g., supply chain costs, demand elasticity).
* **Cross-Domain Synthesis:** Combining quantitative simulations with qualitative market research to provide a 360-degree risk assessment.
* **Autonomous Dependency Handling:** Dynamically identifies and implements specialized libraries like `numpy` for vectorization and `scipy` for statistical distributions.

### 🔬 Research & Synthesis
* **Live Exchange Rate Integration:** Real-time currency conversion and sensitivity analysis.
* **Cross-Domain Knowledge Synthesis:** Combining quantitative data (finance) with qualitative cultural trends (e.g., J-Beauty marketing trends in Ginza).
* **Dependency Autonomy:** Automatically identifies, installs, and verifies required Python libraries (`seaborn`, `plotly`, `scikit-learn`, etc.) without manual intervention.

### 🛡️ Resilience & Self-Debugging
* **Binary Header Recovery:** Capable of identifying and fixing encoding issues (e.g., `0x89` PNG headers) in stream outputs.
* **Automated Skill Verification:** Only saves code to the "Skill DB" after a successful, error-free execution.
* **Stateful Memory:** Maintains a continuous chain of thought across multi-turn interactions.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph / LangChain
* **Models:** Groq Llama-3 70B / 8B (Dynamic Fallback Logic: Utilizing 70B for complex architectural planning and 8B for high-speed sub-task execution and meditation loops).
* **Database:** SQLite (Skill Persistence)
* **Infrastructure:** Streamlit / Python Subprocess
* **Visualization:** Matplotlib, Seaborn, Plotly

---
*Developed with the philosophy that AI should not just talk, but execute and learn.*
