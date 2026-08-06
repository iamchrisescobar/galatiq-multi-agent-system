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
