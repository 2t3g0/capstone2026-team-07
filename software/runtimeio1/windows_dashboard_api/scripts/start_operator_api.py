from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def _load_user_api_key() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    if sys.platform != "win32":
        return

    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "GEMINI_API_KEY")
    except FileNotFoundError:
        return
    if str(value).strip():
        os.environ["GEMINI_API_KEY"] = str(value)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root / "src"))
    _load_user_api_key()
    uvicorn.run(
        "jolgwa_uav.service:app",
        host="0.0.0.0",
        port=9293,
        app_dir=str(project_root / "src"),
    )


if __name__ == "__main__":
    main()
