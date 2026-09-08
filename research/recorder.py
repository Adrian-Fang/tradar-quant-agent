# research/recorder.py，完整文件
"""
自动化因子研究报告记录器。默认开启，通过环境变量 TRADAR_NO_RECORD=1 关闭。

设计原则：记录逻辑内嵌在 research/ 各模块内部（FactorAnalyzer 的
ic_summary/group_diagnostics/group_equity_curves，report.py 的打印函数），
脚本代码不需要任何改动 —— 正常调用这些函数看输出，它们顺手记下同样的信息。

支持多因子并排场景（如 scripts/run_factor.py）：FactorAnalyzer 可传 label 参数
标注"这是哪个因子"，flush() 时按 (phase, factor) 分组，避免多因子对比时
记录混在一起分不清归属。
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

_ENABLED = os.environ.get("TRADAR_NO_RECORD", "0") != "1"
_OUTPUT_DIR = os.environ.get("TRADAR_REPORT_DIR", "data/factor_reports")


class _Session:
    def __init__(self):
        self.entries = []
        self.global_meta = {}
        self._current_phase = None

    @staticmethod
    def _infer_script_name() -> str:
        try:
            if sys.argv and sys.argv[0]:
                path = Path(sys.argv[0])
                if path.stem and path.stem.lower() != "runpy":
                    return path.stem
        except Exception:
            pass
        try:
            import __main__
            if hasattr(__main__, "__file__"):
                return Path(__main__.__file__).stem
        except Exception:
            pass
        return "unknown_script"

    @staticmethod
    def _infer_script_docstring() -> str | None:
        try:
            import __main__
            doc = getattr(__main__, "__doc__", None)
            return doc.strip() if doc else None
        except Exception:
            return None

    def set_phase(self, phase: str | None):
        """标注接下来记录的条目属于哪个阶段（比如 "训练窗口" / "测试窗口(样本外)"）。"""
        self._current_phase = phase

    def set_global_meta(self, **kwargs):
        """
        设置脚本级别的全局meta（universe规则、因子公式、因子清单等），只需设置一次，
        不随每条entry重复。后调用会合并/覆盖同名字段，不是追加。
        """
        if not _ENABLED:
            return
        self.global_meta.update({k: v for k, v in kwargs.items() if v is not None})

    def add(self, entry_type: str, data: dict):
        if not _ENABLED:
            return
        entry = {"type": entry_type, "phase": self._current_phase, **data}
        self.entries.append(entry)

    def flush(self):
        if not _ENABLED or not self.entries:
            return None

        # 按 (phase, factor) 分组聚合成 factor_analysis 条目，支持多因子并排场景
        grouped = {}  # (phase, factor) -> 累积中的分析结果
        backtests, others = [], []

        for e in self.entries:
            phase = e.get("phase")
            factor = e.get("factor")  # ic_summary/group_diagnostics/group_equity 才会带这个字段
            key = (phase, factor)

            if e["type"] == "ic_summary":
                grouped.setdefault(key, {"phase": phase, "factor": factor}).setdefault(
                    "ic_summary", {}
                ).update(e["data"])
            elif e["type"] == "group_diagnostics":
                grouped.setdefault(key, {"phase": phase, "factor": factor})["group_diagnostics"] = e["data"]
            elif e["type"] == "group_equity":
                slot = grouped.setdefault(key, {"phase": phase, "factor": factor})
                slot["group_equity_summary"] = e.get("summary")
                slot["group_monotonicity"] = e.get("monotonicity")
            elif e["type"] == "backtest":
                backtests.append({k: v for k, v in e.items() if k != "type"})
            else:
                others.append(e)

        factor_analyses = list(grouped.values())

        out_dir = Path(_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        script_name = self._infer_script_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out_dir / f"{script_name}_{timestamp}.json"

        report = {
            "script": script_name,
            "script_description": self._infer_script_docstring(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": self.global_meta,
            "factor_analysis": factor_analyses,
            "backtest": backtests,
        }
        if others:
            report["other"] = others

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[recorder] 报告已保存: {filepath}")
        return str(filepath)


_session = _Session()


def get_session() -> _Session:
    return _session


import atexit
atexit.register(_session.flush)