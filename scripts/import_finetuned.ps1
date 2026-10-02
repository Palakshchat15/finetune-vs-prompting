# Import the Colab-trained GGUF into Ollama as the model named in config.yaml (finetune.ollama_name).
# Writes models/Modelfile with the SAME ChatML template and system prompt used in training
# (both are read from src/prompts.py, so they cannot drift), then runs `ollama create`.
#
# Usage (from the project folder):
#   powershell -ExecutionPolicy Bypass -File scripts\import_finetuned.ps1
# Prerequisites: Ollama running; models\<gguf_file> present (unzipped from banking77_outputs.zip).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Project venv missing. Run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt" }

# Model name, GGUF file name, template and system prompt, all from config.yaml / src/prompts.py.
$json = & $py -c "import json; from src.config import load_config; from src.prompts import OLLAMA_TEMPLATE, FINETUNE_SYSTEM; f=load_config()['finetune']; print(json.dumps({'name': f['ollama_name'], 'gguf': f['gguf_file'], 'template': OLLAMA_TEMPLATE, 'system': FINETUNE_SYSTEM}))"
if ($LASTEXITCODE -ne 0) { throw "Could not read config.yaml / src/prompts.py" }
$cfg = $json | ConvertFrom-Json

$gguf = Join-Path $root ("models\" + $cfg.gguf)
if (-not (Test-Path $gguf)) {
    throw "Missing $gguf. Run notebooks/finetune_lora_colab.ipynb on Colab and unzip banking77_outputs.zip into models\."
}
try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null }
catch { throw "Ollama is not reachable at http://localhost:11434. Start the Ollama app and retry." }

# Modelfile: greedy decoding, short output, stop at end of turn. num_ctx small: prompts are ~60 tokens.
$modelfile = Join-Path $root "models\Modelfile"
$lines = @(
    "FROM ./$($cfg.gguf)",
    ('TEMPLATE """' + $cfg.template + '"""'),
    ('SYSTEM """' + $cfg.system + '"""'),
    "PARAMETER temperature 0",
    "PARAMETER num_ctx 1024",
    "PARAMETER num_predict 24",
    'PARAMETER stop "<|im_end|>"',
    'PARAMETER stop "<|im_start|>"'
)
# UTF-8 without BOM (Ollama's parser does not expect a BOM).
[System.IO.File]::WriteAllText($modelfile, ($lines -join "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Wrote $modelfile"

Push-Location (Join-Path $root "models")
try {
    & ollama create $cfg.name -f Modelfile
    if ($LASTEXITCODE -ne 0) { throw "ollama create failed" }
} finally { Pop-Location }

# Smoke test: one message, should print one of the 77 labels (e.g. card_arrival).
$body = @{ model = $cfg.name; stream = $false; messages = @(@{ role = "system"; content = $cfg.system }, @{ role = "user"; content = "My new card still hasn't arrived, when will I get it?" }) } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod -Uri "http://localhost:11434/api/chat" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
Write-Host ("Smoke test prediction: " + $r.message.content)
Write-Host "Next: .venv\Scripts\python scripts\eval_finetuned.py"
