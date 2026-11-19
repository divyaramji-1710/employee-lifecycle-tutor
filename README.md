# Employee Lifecycle Tutor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/uk2jg77mp7/badge.svg)](https://ask.thili.ai/b/uk2jg77mp7)

The Employee Lifecycle Tutor is an interactive tool designed for new HR personnel. It explains common employee lifecycle event sequences and predicts likely next steps. This is achieved by applying frequent itemset mining and language model continued pretraining to anonymized HR event data.

## What you are building

**Goal.** You will have built an interactive tutor that explains common employee lifecycle event sequences and predicts likely next steps for new HR personnel.

**Done when.** The interactive tutor is functional, can explain common HR event sequences, and accurately predicts likely next steps based on the trained language model.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Represent HR events (hiring, training, promotion) as ordered sequences.**  
      Hint: Prepare your HR event data as ordered sequences for analysis.
- [ ] **2. Mine frequent HR event sequences using `thilimarketbasket`.**  
      Hint: Use `lfretail-lfmarketbasket` to find common event sequences.
- [ ] **3. Train an LM to predict the next likely HR event in a sequence.**  
      Hint: Apply language model continued pretraining to predict subsequent events.
- [ ] **4. Design a text-based interactive tutor that queries the LM for explanations or next steps.**  
      Hint: Develop a text-based interface to interact with your trained language model.
- [ ] **5. Evaluate the tutor's ability to accurately guide users through HR processes.**  
      Hint: Design a method to assess how well the tutor guides users through HR processes.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the tutor to provide more detailed explanations for each event, beyond just the next step.
- Integrate a feedback mechanism for users to rate the tutor's helpfulness and accuracy.
- Expand the tutor to cover edge cases or less common HR event sequences, providing guidance for unusual paths.
