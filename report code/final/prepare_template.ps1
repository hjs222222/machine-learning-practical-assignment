# 2026-11-05 最终报告 第1次修改版；保留原教师文件，转换独立模板副本。
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$sourcePath = Join-Path $projectRoot 'Teacher Resources & Requirements\实验报告模版\机器学习课程实践-实验报告模板-3人组队完成.doc'
$targetPath = Join-Path $PSScriptRoot 'template.docx'
if (Test-Path -LiteralPath $targetPath) { throw '模板副本已存在，请直接使用，禁止静默覆盖。' }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$document = $null
try {
    $document = $word.Documents.Open($sourcePath, $false, $true)
    $document.SaveAs2($targetPath, 16)
} finally {
    if ($document) { $document.Close($false) }
    $word.Quit()
}
