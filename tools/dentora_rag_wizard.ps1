param(
    [string]$RepoPath = "",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-RepoPath {
    param([string]$ExplicitPath)

    if ($ExplicitPath) {
        return (Resolve-Path $ExplicitPath).Path
    }

    $fromScript = Split-Path -Parent $PSScriptRoot
    if ($fromScript -and (Test-Path (Join-Path $fromScript ".git"))) {
        return $fromScript
    }

    $default = Join-Path $env:USERPROFILE "DentalStudyAI"
    if (Test-Path (Join-Path $default ".git")) {
        return $default
    }

    return $default
}

function Set-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Value
    )

    $lines = @()
    if (Test-Path $Path) {
        $lines = Get-Content -LiteralPath $Path
    }

    $escapedName = [regex]::Escape($Name)
    $matched = $false
    $output = foreach ($line in $lines) {
        if ($line -match "^\s*$escapedName\s*=") {
            $matched = $true
            "$Name=$Value"
        } else {
            $line
        }
    }

    if (-not $matched) {
        $output += "$Name=$Value"
    }

    Set-Content -LiteralPath $Path -Value $output -Encoding Ascii
}

function Get-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path $Path)) {
        return ""
    }

    $escapedName = [regex]::Escape($Name)
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^\s*$escapedName\s*=\s*(.*)$") {
            return $Matches[1].Trim()
        }
    }

    return ""
}

function Read-SecretPlainText {
    param([string]$Prompt)

    $secure = Read-Host $Prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)

    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

function Select-PdfFile {
    Add-Type -AssemblyName System.Windows.Forms

    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Choose the PDF to add to Dentora RAG"
    $dialog.Filter = "PDF files (*.pdf)|*.pdf"
    $dialog.Multiselect = $false

    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.FileName
    }

    return ""
}

function Select-Category {
    Write-Host ""
    Write-Host "Choose document category:" -ForegroundColor White
    Write-Host "  1. Books"
    Write-Host "  2. Past Papers"
    Write-Host "  3. Viva"
    Write-Host "  4. OSCE"
    Write-Host "  5. Notes"
    Write-Host "  6. Other"

    $choice = Read-Host "Category number [1]"

    switch ($choice) {
        "2" { return "Past Papers" }
        "3" { return "Viva" }
        "4" { return "OSCE" }
        "5" { return "Notes" }
        "6" { return "Other" }
        default { return "Books" }
    }
}

Write-Host ""
Write-Host "Dentora RAG Setup & Indexing Wizard" -ForegroundColor Green
Write-Host "-----------------------------------" -ForegroundColor DarkGray

$repo = Get-RepoPath -ExplicitPath $RepoPath

if (-not (Test-Path (Join-Path $repo ".git"))) {
    Write-Step "DentalStudyAI repo was not found. Cloning it now..."
    git clone "https://github.com/drmes-dev/DentalStudyAI.git" $repo
}

Write-Step "Updating Dentora from GitHub"
git -C $repo pull --ff-only origin main

$venvPython = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating Python virtual environment"

    $created = $false

    try {
        py -3 -m venv (Join-Path $repo ".venv")
        $created = $true
    }
    catch {
    }

    if (-not $created) {
        python -m venv (Join-Path $repo ".venv")
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Could not create or find .venv Python."
}

if (-not $SkipInstall) {
    Write-Step "Installing/updating Dentora dependencies"
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $repo "requirements.txt")
}

$envFile = Join-Path $repo ".env"

$hasPineconeKey = $false
if (Test-Path $envFile) {
    $hasPineconeKey = Select-String -LiteralPath $envFile -Pattern '^\s*PINECONE_API_KEY\s*=\s*\S+' -Quiet
}

if ($hasPineconeKey) {
    Write-Step "Existing Pinecone key found in local .env"
    Write-Host "Reusing it automatically. No key entry is needed." -ForegroundColor Green
}
else {
    Write-Step "Connecting this PC to your Pinecone project"
    Write-Host "No usable Pinecone key was found in the local .env file." -ForegroundColor Yellow
    Write-Host "Paste it once into the hidden prompt; it stays only on this PC." -ForegroundColor DarkGray

    $key = Read-SecretPlainText -Prompt "Pinecone API key"

    if (-not $key) {
        throw "No Pinecone API key was entered."
    }

    Set-DotEnvValue -Path $envFile -Name "PINECONE_API_KEY" -Value $key
}

Set-DotEnvValue -Path $envFile -Name "PINECONE_INDEX_NAME" -Value "dentora-knowledge"
Set-DotEnvValue -Path $envFile -Name "PINECONE_CLOUD" -Value "aws"
Set-DotEnvValue -Path $envFile -Name "PINECONE_REGION" -Value "us-east-1"
Set-DotEnvValue -Path $envFile -Name "PINECONE_EMBED_MODEL" -Value "llama-text-embed-v2"
Set-DotEnvValue -Path $envFile -Name "PINECONE_EMBED_DIMENSION" -Value "1024"

Write-Step "Pinecone local configuration is ready"

do {
    $pdf = Select-PdfFile

    if (-not $pdf) {
        Write-Host ""
        Write-Host "No PDF selected. Setup is complete; you can run this wizard again later." -ForegroundColor Yellow
        break
    }

    $category = Select-Category

    Write-Step "Indexing $([IO.Path]::GetFileName($pdf)) into Dentora"
    Write-Host "Category: $category" -ForegroundColor DarkGray
    Write-Host "Large/scanned PDFs can take time because OCR is done page-by-page." -ForegroundColor DarkGray

    & $venvPython (Join-Path $repo "tools\index_library.py") $pdf --category $category

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Indexing failed. The window will stay open so you can copy the error." -ForegroundColor Red
        Read-Host "Press Enter to finish"
        exit $LASTEXITCODE
    }

    Write-Step "Checking the live Dentora RAG status"

    try {
        $status = Invoke-RestMethod -Uri "https://dentalstudyai.onrender.com/rag/status" -Method Get -TimeoutSec 90

        Write-Host ("Persistent RAG: " + $status.persistent) -ForegroundColor Green
        Write-Host ("Index: " + $status.index_name) -ForegroundColor Green
        Write-Host ("Total vectors: " + $status.total_vector_count) -ForegroundColor Green
    }
    catch {
        Write-Host "The PDF was indexed, but the live status check could not complete." -ForegroundColor Yellow
        Write-Host $_.Exception.Message -ForegroundColor DarkGray
    }

    Write-Host ""
    $again = Read-Host "Add another PDF now? (y/N)"
}
while ($again -match "^[Yy]$")

Write-Host ""
Write-Host "Dentora RAG setup finished." -ForegroundColor Green
Write-Host "You can close this window." -ForegroundColor DarkGray
