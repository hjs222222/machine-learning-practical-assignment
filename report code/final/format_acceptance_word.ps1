# 2026-11-10 三次验收Word第1次展示版：仅修正表格列宽与跨页，不修改内容。
param([Parameter(Mandatory=$true)][string]$InputDocx)
$path = (Resolve-Path -LiteralPath $InputDocx).Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$document = $null
try {
    $document = $word.Documents.Open($path)
    foreach ($table in $document.Tables) {
        $table.Rows.AllowBreakAcrossPages = $false
        $table.Rows.Item(1).HeadingFormat = $true
        $table.Range.ParagraphFormat.KeepWithNext = $true
        $table.Rows.Last.Range.ParagraphFormat.KeepWithNext = $false
        if ($table.Columns.Count -eq 3) {
            $table.AllowAutoFit = $false
            $table.Columns.Item(1).SetWidth(100, 0)
            $table.Columns.Item(2).SetWidth(55, 0)
            $table.Columns.Item(3).SetWidth(313, 0)
            $table.Range.Font.Size = 10
        }
    }
    $document.Save()
} finally {
    if ($document) { $document.Close($false) }
    $word.Quit()
}
