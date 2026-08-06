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