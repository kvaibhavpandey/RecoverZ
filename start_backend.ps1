Set-Location "$PSScriptRoot\\backend"
if (-not (Test-Path ".venv")) { python -m venv .venv }
& ".\\.venv\\Scripts\\python.exe" -m pip install -r requirements.txt
& ".\\.venv\\Scripts\\python.exe" -m app.seed
& ".\\.venv\\Scripts\\python.exe" -m uvicorn app.main:app --reload --port 8000
