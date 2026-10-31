# -*- coding: utf-8 -*-
"""启动真实结果查看窗口，并只截取本程序创建的窗口，不截取整个桌面。"""
import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import scrolledtext

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def capture_window(hwnd, path):
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
    gdi32.SelectObject.restype = wintypes.HANDLE
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    rectangle = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rectangle))
    width, height = rectangle.right - rectangle.left, rectangle.bottom - rectangle.top
    source = user32.GetWindowDC(hwnd)
    memory = gdi32.CreateCompatibleDC(source)
    bitmap = gdi32.CreateCompatibleBitmap(source, width, height)
    old = gdi32.SelectObject(memory, bitmap)
    try:
        ok = user32.PrintWindow(hwnd, memory, 2)
        if not ok:
            raise RuntimeError("Windows拒绝捕获本程序窗口")
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]
        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(header)
        header.biWidth, header.biHeight = width, -height
        header.biPlanes, header.biBitCount = 1, 32
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                   ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        if not gdi32.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(header), 0):
            raise RuntimeError("读取程序窗口位图失败")
        image = Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1)
        image.save(path)
    finally:
        gdi32.SelectObject(memory, old)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        user32.ReleaseDC(hwnd, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    ctypes.windll.user32.SetProcessDPIAware()
    ctypes.windll.user32.GetParent.argtypes = [wintypes.HWND]
    ctypes.windll.user32.GetParent.restype = wintypes.HWND
    out = ROOT / "验收/Acceptance 3/figures"
    out.mkdir(parents=True, exist_ok=True)
    for task in range(1, 8):
        command = [sys.executable, str(ROOT / "code/acceptance3_evidence.py"),
                   "--run-id", args.run_id, "--task", str(task)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
        window = tk.Tk()
        window.title(f"任务{task}+第三次验收真实程序结果查看器")
        window.geometry("1400x980+30+30")
        tk.Label(window, text=f"任务{task} · 实际程序运行结果", font=("Microsoft YaHei", 18),
                 anchor="w").pack(fill="x", padx=20, pady=12)
        tk.Label(window, text="实时读取正式运行文件并执行结果查看命令；下方为该命令的原始标准输出。",
                 font=("Microsoft YaHei", 10), anchor="w").pack(fill="x", padx=20)
        box = scrolledtext.ScrolledText(window, font=("Consolas", 11), wrap="word")
        box.pack(fill="both", expand=True, padx=20, pady=15)
        box.insert("1.0", result.stdout)
        box.configure(state="disabled")
        window.update_idletasks()
        window.update()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        destination = out / f"任务{task}+真实程序窗口截图.png"
        capture_window(hwnd, destination)
        window.destroy()
        print(destination)


if __name__ == "__main__":
    main()
