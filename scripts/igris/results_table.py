"""IGRIS-C 클립별 평가(처음부터 학습)를 G1 교사와 나란히 표로 만들어 docs/igris(.en).md 에 넣는다.

표는 두 문서의 <!-- igris-table --> ... <!-- /igris-table --> 사이를 통째로 바꾼다.
평가는 프레임 0부터 100 롤아웃, 도메인 랜덤화 끔. 끝난 클립만 들어간다.

    python scripts/igris/results_table.py
"""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WBT = "/home/sehoon/Projects/whole_body_tracking/logs"
# run2_subject4 는 igris_transfer.sh(A)에서, 나머지는 igris_all.sh 에서 평가했다.
# 레퍼런스 v1 = 261003 이전 리타게팅(어깨 뒤집힘·발 관통), v2 = 연속성 항 + 발 가중치 200 으로 다시 만든 것.
# v1 로 학습한 것은 폴더로 구분한다: igris_transfer/(run2 A), igris_all/v1ref/. 14개 전부 v2 로 다시 학습한다(261003).
V1_IN_PLACE = set()
IGRIS = [("run2_subject4", f"{WBT}/igris_transfer/eval_a_29999.json", "v1")]
IGRIS += [(os.path.basename(p)[5:-5], p, "v1") for p in glob.glob(f"{WBT}/igris_all/v1ref/eval_*.json")]
IGRIS += [(os.path.basename(p)[5:-5], p, "v1" if os.path.basename(p)[5:-5] in V1_IN_PLACE else "v2")
          for p in glob.glob(f"{WBT}/igris_all/eval_*.json")]

rows = []
for seq, path, ver in sorted(IGRIS, key=lambda t: (t[0], t[2])):
    i = json.load(open(path))
    g = json.load(open(f"{REPO}/outputs/eval/{seq}.json"))
    # 완주가 0이면 오차는 정의되지 않는다(완주한 롤아웃에서만 잰다). 대신 평균 생존 비율로 어디까지 갔는지 본다.
    f = lambda v, fmt: "—" if v != v else format(v, fmt)
    alive = lambda d: d["mean_alive_frames"] / d["motion_frames"]
    rows.append(f"| {seq} | {ver} | {g['success_rate']:.0%} | {i['success_rate']:.0%} | {alive(g):.0%} | {alive(i):.0%} | "
                f"{f(g['e_mpbpe_mm'], '.0f')} | {f(i['e_mpbpe_mm'], '.0f')} | "
                f"{f(g['e_mpjpe_rad'], '.3f')} | {f(i['e_mpjpe_rad'], '.3f')} |")

HEAD = {"ko": "| 클립 | IGRIS 레퍼런스 | G1 완주 | IGRIS 완주 | G1 평균 생존 | IGRIS 평균 생존 | G1 E_mpbpe (mm) | IGRIS E_mpbpe (mm) | G1 E_mpjpe (rad) | IGRIS E_mpjpe (rad) |",
        "en": "| Clip | IGRIS reference | G1 completion | IGRIS completion | G1 mean survival | IGRIS mean survival | G1 E_mpbpe (mm) | IGRIS E_mpbpe (mm) | G1 E_mpjpe (rad) | IGRIS E_mpjpe (rad) |"}
for lang, path in (("ko", "docs/igris.md"), ("en", "docs/igris.en.md")):
    table = "\n".join([HEAD[lang], "|---|---|---|---|---|---|---|---|---|---|", *rows])
    p = os.path.join(REPO, path)
    s = open(p).read()
    s, n = re.subn(r"<!-- igris-table -->.*?<!-- /igris-table -->",
                   f"<!-- igris-table -->\n{table}\n<!-- /igris-table -->", s, flags=re.S)
    assert n == 1, path
    open(p, "w").write(s)
print(f"{len(rows)} clips")
