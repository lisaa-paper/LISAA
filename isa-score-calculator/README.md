
# ISA Score Calculator

ISA (Information Security Awareness) Score Calculator is an interactive Streamlit web app that benchmarks LLM “contester” models on curated security scenarios and then aggregates ratings from multiple “judge” models. The tool fetches answers via OpenRouter, captures judge scores, and generates sub-focus summaries and exports—all from a single browser session.

## Key Features
- **Scenario evaluation pipeline:** 30 criteria × up to 5 scenarios each (150 total) answered by any OpenRouter-accessible model at a deterministic temperature.
- **Pluggable judge panels:** Use the bundled recommendations or provide any number of custom judge models; the app validates every model before runs.
- **Detailed scoring outputs:** Majority voting per scenario, ISA score per sub-focus area, and downloads for raw answers, judge scores, and final metrics.
- **Scenario explorer & leaderboard:** Browse all criteria/topics and review benchmark scores for dozens of well-known LLMs.
- **Robust error diagnostics:** Failed phases show concise summaries, last in-flight operation, and a downloadable traceback to speed debugging.

## Requirements
- Python 3.10+
- OpenRouter API key with access to the contester/judge models you plan to call
- Excel scenarios file located at `data/Criterinos and 100 Scenarios.xlsx` (the default expected by `config.py`)
- Dependencies listed in `requirements.txt`:
  - Streamlit, OpenAI SDK, Pandas, OpenPyXL, Plotly, NumPy, python-dotenv

## Getting Started
1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd isa-score-calculator
   ```
2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Provide data**
   - Ensure `data/Criterinos and 100 Scenarios.xlsx` exists.  
   - If your file has a different name/location, update `config.SCENARIOS_FILE` or adjust `utils/scenario_loader.py`.
5. **Set up API credentials**
   - Obtain an OpenRouter key: https://openrouter.ai/keys  
   - You can either paste it into the sidebar when running the app or export `OPENROUTER_API_KEY` before launch and reference it through a `.env` file or environment variable.

## Running the App Locally
```bash
streamlit run app.py
```
Then open the provided local URL. In the sidebar:
1. Paste the OpenRouter API key (or use the env var).
2. Enter the contester model name.
3. Select the judge panel (recommended or custom, one per line).
4. Click **Validate & Start ISA Calculation** to run through all three phases.

### What Happens Per Phase
1. **Validation:** Lightweight “ping” requests confirm every model is reachable. Any failure keeps you on the start screen with a clear error message.
2. **Phase 1 – Model Answers:** The contester answers each scenario at temperature `0.00001`. Progress updates display topic/scenario counts.
3. **Phase 2 – Judge Scores:** Each judge receives the scenario, criterion, and contester answer with the security system prompt and must respond with 1/2/3.
4. **Phase 3 – Score Calculation:** Majority vote per scenario → averages per criterion & sub-focus area → overall ISA score. Results include charts and CSV exports.

## Configuration Tips
- `config.py` contains the OpenRouter base URL, temperature/max token settings, default judge list, judge system prompt, and category mappings.
- To add more sub-focus areas or rename them, update `config.CATEGORIES`, `config.CATEGORY_NAMES`, and `config.CATEGORY_RANGES`.
- You can safely customize the recommended judge list (`RECOMMENDED_JUDGE_MODELS` in `app.py`) without touching the validation logic.

## Deployment Options
| Option | Notes |
|--------|-------|
| **Streamlit Community Cloud** | Push the repo to GitHub → https://streamlit.io/cloud → point to `app.py` → store secrets via Streamlit’s UI. Great for quick sharing. |
| **Docker + Cloud Run/App Runner/etc.** | Add a `Dockerfile` that installs `requirements.txt` and runs `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`. Deploy the container to your preferred managed service. |
| **Self-hosted VM** | Provision a VM, install deps, set env vars, and run Streamlit under a supervisor (systemd/pm2). Optionally front with Nginx or Caddy for TLS. |

Regardless of platform, remember that each execution triggers many OpenRouter requests (150 answers + 150×N judges), so allocate adequate quotas and consider requiring per-user API keys if hosting publicly.

## Troubleshooting
- **“Model validation failed”**: Verify spelling/access for the contester and every judge; make sure each is available via your OpenRouter account.
- **Scenario file errors**: Check the Excel file path/name and ensure the columns match the expected schema (Criterion ID, Criterion, Scenario No. 1-5). The loader normalizes many variations, but missing data will reduce evaluations.
- **Phase stalls/failures**: Use the inline error summary plus downloadable log to inspect stack traces, HTTP errors, or specific topic/scenario where the call failed.
- **Caching issues**: The scenario workbook is cached via `st.cache_data`. If you modify the Excel file, click “Clear cache” in Streamlit’s hamburger menu or restart the app.


