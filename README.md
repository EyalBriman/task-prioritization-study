# Everyday Task Prioritization Study

This repository contains a Streamlit web application for a small-scale
human–AI interaction study on everyday task prioritization.

Participants interact with the same AI system under three conditions
that differ **only in the level of human control**:

1. **Advisory** – the AI proposes a ranking; the user decides  
2. **Semi-automated** – the AI proposes a ranking; the user may modify it  
3. **Fully automated** – the AI determines the ranking without user input  

Across all conditions, the AI applies a **fixed rubric** based on:
- urgency,
- importance,
- and deadline proximity.

The study measures trust, perceived responsibility, engagement, and
behavioral differences across control levels.

---

## Requirements

- Python 3.9+
- A Groq API key (free tier is sufficient)
- Internet connection (for API calls)

---

## Installation (Local Run)

1. Clone the repository:
   ```bash
   git clone https://github.com/EyalBriman/task-prioritization-study.git
   cd task-prioritization-study
