# Prompt Injection Tester

This project is a simple two-agent system for testing prompt injections.  
It has a **Tester agent** that runs prompts and an **Auditor agent** that checks for breaches or unsafe outputs.
So far the implementation is oriented around Azure deployments. 

---

## Setup

### 1. Create the `.env` file
1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
2. Fill in your credentials

### 2. Install dependencies
1. Create venv
```bash
python -m venv .venv
```
2. Activate the venv
```bash
.venv\Scripts\activate
```
3. Install the requirements
```bash
pip install -r requirements.txt
```
### 3. Prepare your dataset
1. Place your raw dataset (CSV) in the `DatasetUnclean/` folder
2. run the cleaning script
```bash
python pandasCleaner.py
```
3. This will create `Datasets/cleaned_prompts.csv` containing a single column called `prompt`
### 4. Run the project
```bash
python src/main.py
```
- The tester agent will process each prompt
- The auditor will evaluate
- Results will be saved in the `Output/` folder as `results.csv`
#### Debug Feature

In `src/utils.py` there is a debug feature which you can use to limit the amount of prompts being tested.  
It is automatically set to `TRUE` and the limit is set to 10 prompts.
`DEBUG` also allows you to see the amount of tokens consumed per prompt as well as see the total consumption at the end of the run.

- To turn it off, set `DEBUG` to `False`.  
- To change the amount of prompts being tested in debug mode, edit the `DEBUG_LIMIT` value.