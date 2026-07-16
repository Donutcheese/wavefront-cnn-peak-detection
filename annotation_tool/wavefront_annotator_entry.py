"""PyInstaller 打包入口：以包方式导入，保证相对导入与运行时依赖均可用。

开发态仍推荐: python -m wavefront_annotator
"""

from __future__ import annotations

from wavefront_annotator.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
