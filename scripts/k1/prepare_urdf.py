"""ROBOTIS AI Sapiens K1 URDF 에서 IsaacLab 학습용 URDF 를 만든다.

원 모델: github.com/ROBOTIS-GIT/ai_sapiens (Apache-2.0) 의 ai_sapiens_description/urdf/k1_rev1/k1.urdf.
IGRIS 와 달리 토크·속도 한계(96.9 Nm/11.5 rad/s, 47.3 Nm/20.9 rad/s)가 URDF 에 들어 있고 루트가 pelvis 라
바꿀 것은 package:// 메시 경로를 절대 경로로 푸는 것뿐이다. 23 자유도(머리는 fixed).

    python scripts/k1/prepare_urdf.py <ai_sapiens 클론> <출력 urdf>
"""
import sys
from pathlib import Path

src_root = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2])
desc = src_root / "ai_sapiens_description"
text = (desc / "urdf/k1_rev1/k1.urdf").read_text().replace("package://ai_sapiens_description/", f"{desc}/")
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text)
assert "package://" not in text
print(out, text.count("<joint"), "joints")
