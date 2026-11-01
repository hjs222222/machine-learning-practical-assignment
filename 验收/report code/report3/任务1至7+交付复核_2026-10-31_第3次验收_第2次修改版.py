# -*- coding: utf-8 -*-
"""只读复核第二轮交付报告；预览缩略图留在report3/tmp。"""
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
from docx import Document
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '验收/Acceptance 3/第二轮修订版'
RUN = ROOT / 'output/acceptance3_runs/formal_20261031_v2'
DOC = OUT / '任务1至7+第三次阶段验收报告_第2次修改版.docx'
ARCHIVE = ROOT / '验收/任务1至7+第三次阶段验收交付包_第2次修改版.zip'
TASKS = [(2, 'direct', 'LSTM'), (3, 'multi', 'LSTM'), (4, 'direct', 'Mean'),
         (5, 'direct', 'Stacking'), (6, 'multi', 'Mean'), (7, 'multi', 'Stacking')]
report = Document(DOC)
metrics = pd.read_csv(RUN / 'metrics.csv')
tables = [t for t in report.tables if [c.text for c in t.rows[0].cells] == ['h', 'R²', 'RMSE', 'MAE', 'n']]
assert len(tables) == 6
for table, (task, strategy, model) in zip(tables, TASKS):
    values = np.array([[float(c.text) for c in row.cells] for row in table.rows[1:]])
    expected = metrics[(metrics.strategy == strategy) & (metrics.model == model)].sort_values('h')[['h', 'R2', 'RMSE', 'MAE', 'n']].to_numpy()
    assert values.shape == (7, 5)
    np.testing.assert_allclose(values, expected, atol=.000051, rtol=0)
    with Image.open(OUT / f'figures/任务{task}+第二轮真实程序窗口截图.png') as image:
        image.verify()
all_files = sorted(p for p in OUT.rglob('*') if p.is_file())
assert not any(p.suffix.lower() in ['.py', '.pyc', '.log'] for p in all_files)
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    files = z.namelist()
    assert len(files) == 54
    assert not any('填写模板' in p or p.endswith(('.py', '.log', '.pyc')) for p in files)
# 统一预览图用于查看其余过程线、源码摘录与截图，无需改写任何最终交付文件。
images = sorted((OUT / 'figures').glob('*.png'))
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
preview = Image.new('RGB', (1800, 370 * ((len(images) + 2) // 3)), '#fcfcfb')
draw = ImageDraw.Draw(preview)
for i, path in enumerate(images):
    with Image.open(path) as image:
        image.thumbnail((570, 330))
        x, y = (i % 3) * 600 + 15, (i // 3) * 370 + 30
        preview.paste(image.convert('RGB'), (x, y))
        draw.text((x, y - 25), path.stem[:30], fill='#0b0b0b', font=font)
tmp = Path(__file__).parent / 'tmp'
tmp.mkdir(exist_ok=True)
preview_path = tmp / '任务1至7+第二轮全部图表检查缩略图.png'
preview.save(preview_path)
result = dict(word_main_metric_tables=6, word_main_metric_rows=42,
    word_metric_values_match='passed_at_display_precision',
    zip_files=len(files), zip_bytes=ARCHIVE.stat().st_size,
    image_files=len(images), final_directory_files=len(all_files),
    no_sources_logs_templates_in_package=True,
    word_desktop_visual_check='not_performed', preview=str(preview_path))
print(json.dumps(result, ensure_ascii=False, indent=2))
