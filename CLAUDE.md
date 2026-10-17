# CLAUDE.md — Bridgeway ReAct agent (EECE 503P – C4)

University assignment **C4 – Build a Custom ReAct Agent with Configurable Personas**, built on the C3 project
(Bridgeway – Fundraising & Communications Consultancy for NGOs). The owner is a beginner: build step by step,
explain in simple terms, and wait for "next" between steps.

## Ground rules
- **LangGraph only** for the agent. No `AgentExecutor`, `create_react_agent` or prebuilt `ToolNode`: the ReAct loop is
  hand-written (`agent` node, `tools` node, conditional edge `should_continue`).
- Same LLM setup as C3 (`llm.py`, `.env`: Gemini via the OpenAI-compatible endpoint, or OpenAI). Never print or commit keys.
- Business facts and prices live in `business_data.py` and `me/`; never invent others.
- **Git:** show the full commit message and get approval before committing. No AI attribution in commits or PRs.
- Windows: interpreter `py`, venv in `venv/`.

## Structure
- `c4_react_agent.ipynb` — main deliverable: setup, tools, ReAct format, parser, LangGraph loop, personas
  (Layla, Dr. Karam), Instructions block, experiments (`run_all_experiments()` → `logs/experiments.csv`),
  reflection (owner writes the answers), Gradio demo.
- `tools.py` — `search_business_info`, `record_customer_interest`, `record_feedback`, `estimate_campaign_package`.
- C3 files kept for reference: `business_agent.ipynb`, `app.py`, `agent.py`, `router.py`, `reflection.py`, `tests/`
  (`py -m pytest` must keep passing).
