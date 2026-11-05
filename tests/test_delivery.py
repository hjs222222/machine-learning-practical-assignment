import sys
from pathlib import Path
import zipfile
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'report code'/'final'))
from build_final_report import check_links, check_code_zip


def test_relative_links_and_missing_attachment(tmp_path):
    report=tmp_path/'report.md'
    (tmp_path/'data.csv').write_text('a,b\n1,2')
    report.write_text('[表](data.csv)',encoding='utf-8')
    assert check_links(report)==1
    report.write_text('[表](missing.csv)',encoding='utf-8')
    with pytest.raises(ValueError):
        check_links(report)
    report.write_text('[外部路径](../escape.csv)',encoding='utf-8')
    with pytest.raises(ValueError):
        check_links(report)


def test_zip_rejects_data_weights_and_cache(tmp_path):
    valid=tmp_path/'valid.zip'
    with zipfile.ZipFile(valid,'w') as z:
        z.writestr('code/run_all.py','print(1)')
        z.writestr('requirements.txt','numpy')
    check_code_zip(valid)
    for forbidden in ('01047000.csv','weights.pt','code/__pycache__/a.pyc'):
        bad=tmp_path/'bad.zip'
        with zipfile.ZipFile(bad,'w') as z:
            z.writestr(forbidden,'bad')
        with pytest.raises(ValueError):
            check_code_zip(bad)
