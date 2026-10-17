"""점프 구간 채점표: 레퍼런스 후보들을 G1(완주 100%)과 같은 잣대로 한 줄씩.
    python scripts/igris/jump_scorecard.py robot:pkl폴더 ...   (GMR 환경, 저장소 루트)
dance2 130-140 s: 0.3 s 이상 체공 횟수(사람 1, G1 1), 골반 최고(+m, 서 있는 높이 대비), 발 최고(+m);
클립 전체: 관통 p1(cm), 두 발 공중 비율, 몸통 허우적 프레임(ref_smoothness 기준). jumps1 224-240 s 최대 체공(s).
E31/P10 실패(10/17) 뒤 만든 것: 점프 전후 발이 뜨는 횟수와 점프 부풀림이 완주를 가르는 수치였다.
"""
import sys, os, pickle, numpy as np, mujoco
from general_motion_retargeting import ROBOT_XML_DICT
sys.argv, specs = sys.argv[:1], sys.argv[1:]
ns = {}; exec(open("scripts/igris/ref_smoothness.py").read().split("\ndirs = ")[0], ns)
from general_motion_retargeting.utils.lafan1 import load_bvh_file

def footlow(m, d, q):
    feet = [g for g in range(m.ngeom) if m.geom_contype[g] and any(s in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for s in ("ankle_roll", "foot"))]
    out = np.empty(len(q))
    for k in range(len(q)):
        d.qpos[:] = q[k]; mujoco.mj_kinematics(m, d)
        out[k] = min(d.geom_xpos[g][2] - (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][0]) for g in feet)
    return out

def flights(up, lo, hi, fps=30):
    seg = np.flatnonzero(up[lo * fps:hi * fps]); cuts = [c for c in np.split(seg, np.flatnonzero(np.diff(seg) > 1) + 1) if len(c)] if len(seg) else []
    return [len(c) / fps for c in cuts]

HB = {}
for seq in ("dance2_subject3", "jumps1_subject1"):
    HB[seq] = load_bvh_file(f"/home/sehoon/data/lafan1/{seq}.bvh", format="lafan1")[0]
hfoot = np.array([min(f[k][0][2] for k in ("LeftFoot", "RightFoot", "LeftToe", "RightToe")) for f in HB["dance2_subject3"]]); hg = np.percentile(hfoot, 5)
hup = hfoot > hg + 0.03; hhip = np.array([f["Hips"][0][2] for f in HB["dance2_subject3"]])
print(f"{'human':22s} 체공≥0.3s {sum(f >= 0.3 for f in flights(hup, 130, 140))}  골반 +{hhip[130*30:140*30].max() - np.median(hhip):.2f}m")
for spec in [("unitree_g1", "outputs/retarget")] + [tuple(s.split(":", 1)) for s in specs]:
    robot, dd = spec; m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot])); d = mujoco.MjData(m)
    r = pickle.load(open(f"{dd}/dance2_subject3.pkl", "rb")); q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    fz = footlow(m, d, q); g0 = np.percentile(fz, 5); up = fz > g0 + 0.03; z = q[:, 2]; w = slice(130 * 30, 140 * 30)
    pr = ns["pairs"](robot); links = [l for _, l in pr]
    human = np.stack([ns["ang_speed"](np.array([f[b][1] for f in HB["dance2_subject3"]])) for b, _ in pr], 1)
    ex = ns["robot_speeds"](robot, f"{dd}/dance2_subject3.pkl", links)[:len(human)] - human[:len(q) - 1]
    trunk = [i for i, l in enumerate(links) if l in ns["TRUNK"]]; flail = int((ex[:, trunk] > 300).any(1).sum())
    air = "-"
    if os.path.exists(f"{dd}/jumps1_subject1.pkl"):
        r2 = pickle.load(open(f"{dd}/jumps1_subject1.pkl", "rb")); q2 = np.concatenate([r2["root_pos"], r2["root_rot"][:, [3, 0, 1, 2]], r2["dof_pos"]], 1)
        f2 = footlow(m, d, q2); up2 = f2 > np.percentile(f2, 5) + 0.03; air = f"{max(flights(up2, 224, 240) + [0]):.2f}"
    print(f"{os.path.basename(dd.rstrip('/')):22s} 체공≥0.3s {sum(f >= 0.3 for f in flights(up, 130, 140))}  골반 +{z[w].max() - np.median(z):.2f}m  발 +{fz[w].max() - g0:.2f}m | 관통p1 {np.percentile(fz - g0, 1) * 100:+.1f}cm 공중비율 {up.mean():.3f} 몸통허우적 {flail:3d} | jumps1 체공 {air}s")
