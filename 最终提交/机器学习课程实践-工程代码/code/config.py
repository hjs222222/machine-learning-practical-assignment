"""本次新实验的唯一数据源与运行清单。"""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'Teacher Resources & Requirements' / '01047000.csv'
SEED = 42


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def new_run_dir(data_path, destination, config):
    if Path(data_path).resolve() != DATA_PATH.resolve():
        raise ValueError('只允许本项目教师 CSV 路径')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('运行目录已存在，禁止覆盖或热启动')
    destination.mkdir(parents=True)
    sha = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    git = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True)
    save_json(destination / 'manifest.json', dict(status='created', created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        data_sha256=digest(DATA_PATH), config_sha256=sha, config=config, python=platform.python_version(),
        git_commit=git.stdout.strip(), seed=SEED, expected_base_models=24, expected_ensemble_models=16,
        test_limitation='2004已在旧项目中查看；本次为预注册配置后的回溯评估，不是全新盲测'))
    return destination
