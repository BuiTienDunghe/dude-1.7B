"""In thông tin gói PyPI đã tải về (results/baselines/pypi/<pkg>.json): phiên bản, license, phụ thuộc, các bản phát hành."""
import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    i = d["info"]
    print("==", i["name"], i["version"], "| license:", i["license"], "| python:", i["requires_python"])
    print("   urls:", i.get("project_urls") or i.get("home_page"))
    print("   requires:", i["requires_dist"])
    print("   releases:", list(d["releases"])[-10:])
    print("   files:", [(u["filename"], u["upload_time"], u["digests"]["sha256"][:16]) for u in d["urls"]])
