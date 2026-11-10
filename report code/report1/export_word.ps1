# 2026-11-10 第1次验收 Word展示版，第1次修改版；保留MD原文和附件相对路径。
$folder = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../验收/acceptance 1'))
$source = Join-Path $folder '任务1至8+第1次阶段验收记录.md'
$target = [IO.Path]::ChangeExtension($source, '.docx')
if (Test-Path -LiteralPath $target) { throw 'Word展示版已存在，禁止静默覆盖' }
pandoc $source --from=markdown --to=docx --resource-path=$folder --output=$target
if ($LASTEXITCODE -ne 0) { throw 'Word转换失败' }
& (Join-Path $PSScriptRoot '../final/format_acceptance_word.ps1') -InputDocx $target
