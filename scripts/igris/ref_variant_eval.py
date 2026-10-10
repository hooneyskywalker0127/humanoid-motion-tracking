"""리타게팅 설정 실험(outputs/retarget_igris_exp/*)을 사람·G1 과 같은 잣대로 잰다.

    python scripts/igris/ref_variant_eval.py outputs/retarget_igris outputs/retarget_igris_exp/E1_scale095 robotis_k1:outputs/retarget_k1 ...

인자는 pkl 폴더. `로봇:폴더` 로 로봇을 지정할 수 있고(기본 igris_c), 클립은 마지막 폴더의 pkl 로 정한다.

클립마다: 두 발이 모두 바닥 3cm 위에 뜬 프레임 비율(float), 발 관통 p1(cm), 서 있는 첫 3초 hip_roll(°),
골반 최저 높이(서 있는 높이 대비), 점프 체공(두 발 공중 구간 중 가장 긴 것, 초). 사람 BVH 와 G1 레퍼런스도 같은 줄에.
"""
import pickle, sys, glob, os
import numpy as np, mujoco
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

FPS = 30
FEET = {"igris_c": (("l_foot_original", "r_foot_original"), ("l_hip_roll", "r_hip_roll")),
        "robotis_k1": (("left_ankle_roll_link", "right_ankle_roll_link"), ("left_hip_roll_joint", "right_hip_roll_joint")),
        "unitree_g1": (("left_ankle_roll_link", "right_ankle_roll_link"), ("left_hip_roll_joint", "right_hip_roll_joint"))}


def airtime(both_up):
    """연속 True 구간 길이 중 최대(초)"""
    best = run = 0
    for b in both_up:
        run = run + 1 if b else 0
        best = max(best, run)
    return best / FPS


def robot_stats(robot, pkl):
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot])); d = mujoco.MjData(m)
    r = pickle.load(open(pkl, "rb"))
    q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    feet, rolls = FEET[robot]
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    hr = [7 + jn.index(h) for h in rolls]
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in feet]
    geoms = {b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]] for b in fb}
    lows = []
    for t in range(len(q)):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(m, d)
        lows.append([min(d.geom_xpos[g][2] - (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][0])
                         for g in geoms[b]) for b in fb])
    lows = np.array(lows); ground = np.percentile(lows.min(1), 5); both = lows.min(1) > ground + 0.03
    z = q[:, 2]
    return dict(float=both.mean(), pen=np.percentile(lows.min(1) - ground, 1) * 100, hiproll=np.degrees(np.abs(q[:90, hr]).mean()),
                rootmin=z.min() / np.median(z), air=airtime(both))


def human_stats(seq):
    fr, _ = load_bvh_file(f"/home/sehoon/data/lafan1/{seq}.bvh", format="lafan1")
    hips = np.array([f["Hips"][0][2] for f in fr])
    lows = np.array([[min(f[a][0][2], f[b][0][2]) for a, b in (("LeftFoot", "LeftToe"), ("RightFoot", "RightToe"))] for f in fr])
    ground = np.percentile(lows.min(1), 5); both = lows.min(1) > ground + 0.03
    return dict(float=both.mean(), pen=float("nan"), hiproll=float("nan"), rootmin=hips.min() / np.median(hips), air=airtime(both))


fmt = lambda s: f"float {s['float']:.3f}  pen {s['pen']:+5.1f}cm  hiproll {s['hiproll']:4.1f}°  rootmin {s['rootmin']:.2f}  air {s['air']:.2f}s"
dirs = [(a.split(":", 1) if ":" in a else ("igris_c", a)) for a in sys.argv[1:]]
seqs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{dirs[-1][1]}/*.pkl"))
for seq in seqs:
    print(f"== {seq}")
    print(f"  {'human':14s} {fmt(human_stats(seq))}")
    print(f"  {'G1':14s} {fmt(robot_stats('unitree_g1', f'outputs/retarget/{seq}.pkl'))}")
    for robot, dd in dirs:
        p = f"{dd}/{seq}.pkl"
        if os.path.exists(p):
            print(f"  {os.path.basename(dd.rstrip('/')):14s} {fmt(robot_stats(robot, p))}")
