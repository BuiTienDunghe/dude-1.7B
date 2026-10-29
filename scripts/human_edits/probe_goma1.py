"""Đã gộp vào probe_user_item.py (nhận đường dẫn JSON bất kỳ). Giữ để lệnh cũ còn chạy:
probe_goma1.py build|table [N]  ==  probe_user_item.py build|table results/bench_v1_2_trial/user_revised_probe/gomaN_user.json"""
import runpy
import sys
from pathlib import Path

N = sys.argv[2] if len(sys.argv) > 2 else "1"
assert N in ("1", "2")
H = Path(__file__).resolve().parent
sys.argv = [str(H / "probe_user_item.py"), sys.argv[1], f"results/bench_v1_2_trial/user_revised_probe/goma{N}_user.json"]
runpy.run_path(sys.argv[0], run_name="__main__")
