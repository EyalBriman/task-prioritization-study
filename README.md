# Everyday Task Prioritization Study

This repository contains a Streamlit web application for a small-scale
human–AI interaction study on everyday task prioritization.

Participants interact with the same language model under three conditions
that differ only in the level of human control:
1. Advisory (AI proposes, user decides)
2. Semi-automated (AI proposes, user may modify)
3. Fully automated (AI decides, no user input)

The AI applies a fixed rubric based on urgency, importance, and deadlines
across all conditions.

---

## Requirements
- Python 3.9+
- A Gemini API key (free tier is sufficient)

---

## Run locally

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
