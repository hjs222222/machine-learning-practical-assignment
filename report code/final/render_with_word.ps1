# 2026-11-05 最终报告 第1次修改版；Word原生排版更新与PDF导出。
param([Parameter(Mandatory=$true)][string]$InputDocx,
      [Parameter(Mandatory=$true)][string]$OutputDirectory)
$sourcePath = (Resolve-Path -LiteralPath $InputDocx).Path
$targetDir = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$document = $null
try {
    $document = $word.Documents.Open($sourcePath, $false, $true)
    $document.Fields.Update() | Out-Null
    foreach ($section in $document.Sections) {
        foreach ($footer in $section.Footers) { $footer.Range.Fields.Update() | Out-Null }
    }
    $document.Repaginate()
    $document.ExportAsFixedFormat((Join-Path $targetDir 'report.pdf'), 17)
} finally {
    if ($document) { $document.Close($false) }
    $word.Quit()
}
pdftoppm -r 145 -png (Join-Path $targetDir 'report.pdf') (Join-Path $targetDir 'page')
if ($LASTEXITCODE -ne 0) { throw 'PDF页面光栅化失败' }
