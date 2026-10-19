"""새 로봇 레퍼런스 채점표: 리타게팅 결과를 G1(14클립 완주 95% 이상)과 같은 잣대로 재서 통과/탈락을 찍는다.

    python scripts/newrobot/scorecard.py robot:pkl폴더 [robot:pkl폴더 ...] [--g1 outputs/retarget]   (GMR 환경, 저장소 루트)

클립마다 한 줄. 문헌의 절대 문턱(PHUMA 관절 98%, NMR 관절 점프 0.5 rad, 비부유 1 cm)은 완주 100% 인 G1 레퍼런스 자체가
넘기므로(dance2 관절 점프 72프레임, run2 231프레임, waist_pitch 포화 54%) 모든 항목을 **G1 과의 차이**로 재고, 문턱은
10/17 탈락 레퍼런스(IGRIS E31·K1 P10, dance2 0%)와 통과 레퍼런스(IGRIS v4·K1 v1, dance2 100%)가 갈리도록 잡았다:
  limit  관절 범위 98% 를 넘는 (프레임,관절) 비율 − G1                 ≤ +5 %p
  jump   한 프레임에 0.5 rad 넘게 뛰는 관절이 있는 프레임 수 − G1       ≤ +20
  pen    발 바닥 관통 p1(서 있는 높이 기준) − G1                         ≥ −1 cm
  float  사람 발이 바닥인데 로봇 발이 3 cm 넘게 뜬 프레임 비율 − G1     ≤ +15 %p (탈락 레퍼런스도 +7~8 이라 변별력은 없음, 임시)
  air    두 발이 3 cm 넘게 뜬 프레임 비율 − **사람**(체공 부풀림)        ≤ +0.08   (E31 +0.091, P10 +0.105 / v4 +0.058, v1 −0.002, G1 +0.027)
  apex   골반 최고 높이(서 있는 높이 대비) − **사람**(점프 부풀림)       ≤ +0.05 m (E31 +0.07, P10 +0.08 / v4 −0.11, v1 −0.02, G1 −0.03)
  flail  몸통(pelvis·torso)이 사람보다 300°/s 넘게 빨리 도는 프레임 − G1  ≤ +10
  arm    위팔·아래팔 방향 오차 평균(°) − G1                               ≤ +10 (E12 팔 0자세 prior 는 +60)
  pitch  루트 pitch 가 G1 과 15° 넘게 다른 프레임 비율(골반 좌표계 차이는 중앙값으로 뺀다) ≤ 5 %
  rom    사람 무릎 굽힘이 로봇 무릎 범위를 넘는 프레임 비율(경고만)      > 2 % 면 "무릎!" — 운동학적으로 불가능한 구간, 95% 미만 예상
         (obstacles3: IGRIS-C 131° 4.1% → 81%, K1 135° 3.9% → 94%; dance2·walk 0%)
모두 통과해야 학습한다. 탈락 항목은 보통 make_ik_config.py 의 매핑/오프셋(arm·pitch), 스케일(air·apex·float),
retarget_all.py 의 --contact_weight/--ground_lift(pen·float), --vel_limit/--posture(jump·flail) 로 고친다.
run2·jumps1 문턱은 10/19 학습 결과로 다시 본다(v4 run2 가 air +0.13·float +12 로 찍힘 — 95% 이상 나오면 문턱을 다시 잡는다).
"""
import argparse, glob, os, pickle, sys
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

sys.path.insert(0, "scripts/igris")
ns = {}; exec(open("scripts/igris/ref_smoothness.py").read().split("\ndirs = ")[0], ns)
import ref_arm_error as arm

FPS, HUMAN_FOOT = 30, ("LeftFoot", "RightFoot", "LeftToe", "RightToe")
THR = dict(limit=0.05, jump=20, pen=-1.0, float=0.15, air=0.08, apex=0.05, flail=10, arm=10.0, pitch=0.05)
VS_HUMAN = ("air", "apex")  # 사람 원본과 비교하는 항목(나머지는 G1 과 비교)
ROM_WARN, KNEE = 0.02, ("UpLeg", "Leg", "Foot")


def load(pkl):
    r = pickle.load(open(pkl, "rb"))
    return np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)


def footlow(m, d, q):
    feet = [g for g in range(m.ngeom) if m.geom_contype[g] and any(s in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for s in ("ankle_roll", "foot"))]
    assert feet, "발 충돌 형상(ankle_roll/foot 몸체)을 못 찾았다"
    out = np.empty(len(q))
    for k in range(len(q)):
        d.qpos[:] = q[k]; mujoco.mj_kinematics(m, d)
        out[k] = min(d.geom_xpos[g][2] - (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][0]) for g in feet)
    return out


def knee_demand(frames, robot):
    """사람 무릎 굽힘각(°, 펴면 0) 최대(좌·우)가 로봇 무릎 가동 범위를 넘는 프레임 비율"""
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    rom = min(np.degrees(m.jnt_range[j, 1] - m.jnt_range[j, 0]) for j in range(m.njnt) if "knee" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).lower())
    def flex(side):
        a, b, c = (np.array([f[side + k][0] for f in frames]) for k in KNEE); u, v = a - b, c - b
        return 180 - np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
    return float((np.maximum(flex("Left"), flex("Right")) > rom).mean()), rom


def measure(robot, pkl, frames, human_contact):
    """채점에 쓰는 원수치. G1 과 상대 비교하는 항목은 (flight 횟수, apex, arm, pitch 배열) 로 돌려준다."""
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot])); d = mujoco.MjData(m); q = load(pkl); n = min(len(q), len(frames))
    dof = q[:, 7:]; lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]; c, w = (lo + hi) / 2, (hi - lo) / 2
    fz = footlow(m, d, q); g0 = np.percentile(fz, 5); rel = fz - g0
    pr = ns["pairs"](robot); links = [l for _, l in pr]
    human = np.stack([ns["ang_speed"](np.array([f[b][1] for f in frames])) for b, _ in pr], 1)
    ex = ns["robot_speeds"](robot, pkl, links)[:n - 1] - human[:n - 1]; trunk = [i for i, l in enumerate(links) if l in ns["TRUNK"]]
    ae = arm.ang(arm.robot_dirs(robot, pkl)[:n], arm.human_dirs(frames)[:n])
    return dict(limit=float((np.abs(dof - c) > 0.98 * w).mean()), jump=int((np.abs(np.diff(dof, axis=0)) > 0.5).any(1).sum()),
                pen=float(np.percentile(rel, 1) * 100), float=float((rel[:n][human_contact[:n]] > 0.03).mean()),
                air=float((rel > 0.03).mean()), apex=float(q[:, 2].max() - np.median(q[:, 2])),
                flail=int((ex[:, trunk] > ns["THR"]).any(1).sum()), arm=float(ae.mean()),
                pitch=R.from_quat(q[:n, 3:7], scalar_first=True).as_euler("ZYX", degrees=True)[:, 1])


def human_air_apex(frames):
    lf = np.array([min(f[k][0][2] for k in ("LeftFoot", "LeftToe")) for f in frames]); rf = np.array([min(f[k][0][2] for k in ("RightFoot", "RightToe")) for f in frames])
    g = np.percentile(np.minimum(lf, rf), 5); hip = np.array([f["Hips"][0][2] for f in frames])
    return dict(air=float(((lf > g + 0.03) & (rf > g + 0.03)).mean()), apex=float(hip.max() - np.median(hip)))


def score(r, g, h):
    n = min(len(r["pitch"]), len(g["pitch"]))
    v = {k: r[k] - (h if k in VS_HUMAN else g)[k] for k in ("limit", "jump", "pen", "float", "air", "apex", "flail", "arm")}
    dp = r["pitch"][:n] - g["pitch"][:n]; v["pitch"] = float((np.abs(dp - np.median(dp)) > 15).mean())
    ok = {k: (v[k] >= THR[k]) if k == "pen" else (v[k] <= THR[k]) for k in v}
    return v, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("specs", nargs="+"); ap.add_argument("--g1", default="outputs/retarget"); ap.add_argument("--bvh", default="/home/sehoon/data/lafan1")
    a = ap.parse_args(); specs = [s.split(":", 1) for s in a.specs]
    fmt = "{:10s}{:>8s}{:>6s}{:>8s}{:>8s}{:>7s}{:>8s}{:>7s}{:>7s}{:>7s}{:>6s}  {}"
    for seq in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{specs[0][1]}/*.pkl")):
        if not os.path.exists(f"{a.g1}/{seq}.pkl"):
            print(f"== {seq}: G1 레퍼런스 없음, 건너뜀"); continue
        frames, _ = load_bvh_file(f"{a.bvh}/{seq}.bvh", format="lafan1")
        hf = np.array([min(f[k][0][2] for k in HUMAN_FOOT) for f in frames]); contact = hf < np.percentile(hf, 5) + 0.03
        g = measure("unitree_g1", f"{a.g1}/{seq}.pkl", frames, contact); h = human_air_apex(frames)
        print(f"== {seq}  (G1: limit {g['limit']*100:.1f}% jump {g['jump']} pen {g['pen']:.1f} float {g['float']*100:.0f}% flail {g['flail']} arm {g['arm']:.1f}° | 사람 air {h['air']:.3f} apex +{h['apex']:.2f})")
        print(fmt.format("", "limit%p", "jump", "pen cm", "float%p", "air", "apex m", "flail", "arm°", "pitch%", "rom%", "판정"))
        for robot, dd in specs:
            if not os.path.exists(f"{dd}/{seq}.pkl"):
                continue
            v, ok = score(measure(robot, f"{dd}/{seq}.pkl", frames, contact), g, h); rom, deg = knee_demand(frames, robot)
            cell = lambda k, s: s + ("" if ok[k] else "!")
            verdict = ("통과" if all(ok.values()) else "탈락 " + ",".join(k for k in ok if not ok[k])) + (f"  무릎! (사람 굽힘 > 로봇 {deg:.0f}°)" if rom > ROM_WARN else "")
            print(fmt.format(os.path.basename(dd.rstrip("/"))[-10:], cell("limit", f"{v['limit']*100:+.1f}"), cell("jump", f"{v['jump']:+d}"), cell("pen", f"{v['pen']:+.1f}"),
                             cell("float", f"{v['float']*100:+.0f}"), cell("air", f"{v['air']:+.3f}"), cell("apex", f"{v['apex']:+.2f}"), cell("flail", f"{v['flail']:+d}"),
                             cell("arm", f"{v['arm']:+.1f}"), cell("pitch", f"{v['pitch']*100:.0f}"), f"{rom*100:.1f}", verdict))
