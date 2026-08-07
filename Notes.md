create virtual environment:
$ python -m venv .venv

activate using Windows/Git Bash
$ source .venv/Scripts/activate

Install required dependencis:
$ pip install -r requirements.txt

Initialize the database:
$ python -m invoice_system.database

Inspect the database:
$ sqlite3 inventory.db "SELECT item, stock FROM inventory;"

To run test suite:
$ python -m pytest -q

note: -q flag for quiet mode, -v flag for verbose Using python -m pytest is preferable to calling pytest directly 
because it ensures pytest uses the same interpreter and includes 
the current project directory in the import path.

To switch providers (Xai, OpenAi, etc), change the environment fields in .env


Ran a smoke test using Grok api:
$ python -m scripts.smoke_test_xai
Provider: xai
Model: grok-build-0.1
Maximum output tokens: 2048

Structured extraction:
{
  "invoice_number": "INV-1001",
  "vendor": "Widgets Inc.",
  "amount": "5000.0",
  "items": [
    {
      "name": "WidgetA",
      "quantity": 10
    },
    {
      "name": "WidgetB",
      "quantity": 5
    }
  ],
  "invoice_date": "2026-01-15",
  "due_date": "2026-02-01"
}

IMPORTANT: validation currently checks inventory line by line, rather than aggregating duplicate products.

For example, INV-1013 contains multiple entries for the same products. The supplied invoice has repeated WidgetA, WidgetB, and GadgetX lines.

change max_revisions = 2  to = 1 in workflow.py?


Current workflow:

Document Loading
       ↓
Ingestion
       ↓
Validation
       ↓
Approval Policy
       ↓
Approval Agent
       ↓
Approval Critic
    ┌──┴───────────────┐
 accept              revise
    │                   │
    │            Approval Agent
    │                   │
    │              Critic again
    │                   │
    │          revisions exhausted?
    │                   │
    │              MANUAL REVIEW
    │                   │
    │        write manual_reviews.jsonl
    │                   │
    │                  END
    ↓
Approval Finalization
       ↓
  final decision
   /          \
approve      reject
  ↓            ↓
Payment      Rejection Handler
  ↓            ↓
payment      write
result       rejections.jsonl
  ↓            ↓
COMPLETED    COMPLETED

Need to work on:
fix line by line validation of items
add entry point (main)
add verbose for entry point
add option to --list-manual-reviews