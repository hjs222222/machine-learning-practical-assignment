# -*- coding: utf-8 -*-
"""2026-11-01，第3次验收第3次修改版：只修正文图排版并复核交付，不训练。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

from docx import Document
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'验收/Acceptance 3/第三轮结果修订版'
NAME = '任务1至7+第三次阶段验收报告_第3次修改版'
EXTRA = '直接与多输出的特征模式、选参及轮数协议不同，因此这里只对照各自冻结策略的结果，不属于同特征、同预算的纯网络结构消融。'


def main():
    source = Path(__file__).with_name('任务1至7+报告生成_2026-11-01_第3次验收_第3次修改版.py')
    spec = importlib.util.spec_from_file_location('report_v3',source)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    word,md = OUT/(NAME+'.docx'),OUT/(NAME+'.md')
    archive = ROOT/'验收/任务1至7+第三次阶段验收交付包_第3次修改版.zip'
    check_path = OUT/'artifacts/任务1至7+第3次修改版交付核验.json'
    document = Document(word)
    text = md.read_text(encoding='utf-8')
    check = json.loads(check_path.read_text(encoding='utf-8'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        members = z.namelist()
    figure = OUT/'figures/任务3及6及7+第二三轮RMSE对照.png'
    old_image = hashlib.sha256(figure.read_bytes()).digest()
    metrics = pd.read_csv(OUT/'data/任务1至7+主seed42完整指标与来源.csv')
    previous = pd.read_csv(report.DIRECT/'metrics.csv')
    report.comparison_plot(previous,metrics,figure)
    replacements = 0
    for relation in document.part.rels.values():
        if relation.reltype.endswith('/image') and hashlib.sha256(relation.target_part.blob).digest()==old_image:
            relation.target_part._blob = figure.read_bytes()
            replacements += 1
    assert replacements==1
    anchor = '元层候选比较也是开发选参成绩，不作统计显著性证明。'
    if EXTRA not in text:
        assert text.count(anchor)==1
        text = text.replace(anchor,anchor+EXTRA)
        found = [p for p in document.paragraphs if anchor in p.text]
        assert len(found)==1
        found[0].add_run(EXTRA)
    document.save(word)
    md.write_text(text,encoding='utf-8')
    loaded = Document(word)
    assert len(loaded.inline_shapes)==16
    tables = [t for t in loaded.tables if [c.text for c in t.rows[0].cells]==['h','R²','RMSE','MAE','n']]
    assert len(tables)==6
    for table,(_,strategy,model,_) in zip(tables,report.v.TASKS):
        expected = metrics[(metrics.strategy==strategy)&(metrics.model==model)].sort_values('h')[['h','R2','RMSE','MAE','n']].to_numpy()
        actual = np.array([[float(c.text) for c in row.cells] for row in table.rows[1:]])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=.000051)
    links = set()
    for relation in loaded.part.rels.values():
        if relation.reltype.endswith('/hyperlink'):
            path = (OUT/unquote(relation.target_ref)).resolve()
            assert path.is_relative_to(OUT.resolve()) and path.is_file()
            links.add(path)
    for ref in re.findall(r'\]\(([^)]+)\)',text):
        path = (OUT/unquote(ref)).resolve()
        assert path.is_relative_to(OUT.resolve()) and path.is_file()
    assert len(links)==56
    assert not any('方案' in p.text or '计划' in p.text for p in loaded.paragraphs if p.style.name.startswith('Heading'))
    assert EXTRA in text and any(EXTRA in p.text for p in loaded.paragraphs)
    check['comparison_endpoint_labels'] = 'separated'
    check['strategy_protocol_limitation'] = 'disclosed'
    check_path.write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary = archive.with_suffix('.checked.zip')
    assert not temporary.exists()
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in members:
            relative = Path(*Path(name).parts[1:])
            path = OUT/relative
            assert path.is_file() and path.suffix.lower() not in ['.py','.pyc','.log']
            assert not any(word in name for word in ['训练方案','训练执行方案'])
            z.write(path,name)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        for name in members:
            path = OUT/Path(*Path(name).parts[1:])
            assert z.read(name)==path.read_bytes()
    temporary.replace(archive)
    with zipfile.ZipFile(word) as z:
        assert z.testzip() is None
    print(json.dumps(check,ensure_ascii=False,indent=2))
    print('ZIP逐文件一致：',len(members))


if __name__=='__main__': main()
