# -*- coding: utf-8 -*-
"""第二轮真实结果窗口截图，2026-10-31，第2次修改版；只捕获本程序窗口。"""
import argparse
import ctypes
from ctypes import wintypes
import importlib.util
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import scrolledtext

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name('任务1至7+运行截图_2026-10-31_第3次验收_第1次修改版.py')
    spec = importlib.util.spec_from_file_location('capture_v1', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctypes.windll.user32.SetProcessDPIAware()
    ctypes.windll.user32.GetParent.argtypes = [wintypes.HWND]
    ctypes.windll.user32.GetParent.restype = wintypes.HWND
    out = ROOT / '验收/Acceptance 3/第二轮修订版/figures'
    out.mkdir(parents=True, exist_ok=True)
    for task in range(1, 8):
        destination = out / f'任务{task}+第二轮真实程序窗口截图.png'
        if destination.exists():
            raise FileExistsError(destination)
        command = [sys.executable, str(ROOT / 'code/acceptance3_v2_evidence.py'),
                   '--run-id', args.run_id, '--task', str(task)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=True)
        window = tk.Tk()
        window.title(f'任务{task}+第二轮正式结果查看器')
        window.geometry('1400x900+20+20')
        tk.Label(window, text=f'任务{task} · 第二轮已核验程序结果', font=('Microsoft YaHei', 17), anchor='w').pack(fill='x', padx=18, pady=10)
        tk.Label(window, text='以下是实际执行结果读取命令的原始stdout；不是训练过程截图。',
                 font=('Microsoft YaHei', 10), anchor='w').pack(fill='x', padx=18)
        box = scrolledtext.ScrolledText(window, font=('Consolas', 11), wrap='none')
        box.pack(fill='both', expand=True, padx=18, pady=12)
        box.insert('1.0', result.stdout)
        box.configure(state='disabled')
        window.update_idletasks()
        window.update()
        module.capture_window(ctypes.windll.user32.GetParent(window.winfo_id()), destination)
        window.destroy()
        print(destination)


if __name__ == '__main__':
    main()
