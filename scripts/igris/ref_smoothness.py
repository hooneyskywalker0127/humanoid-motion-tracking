"""리타게팅 레퍼런스의 허우적거림 검사: 사람 뼈와 짝지어진 로봇 링크의 회전 속도를 프레임마다 비교한다.

    python scripts/igris/ref_smoothness.py robot:pkl폴더 [robot:pkl폴더 ...]   (GMR 환경, 저장소 루트에서)

IK 표(ik_match_table2)에서 회전을 맞추는 (사람 뼈, 로봇 링크) 쌍마다 각속도(°/s)를 구하고, 로봇이 사람보다 THR(°/s)
이상 빠른 프레임을 센다. 몸통(Hips·Spine2)은 사람이 느리게 도는 부위라 따로 센다 — 팔다리 목표가 닿지 않을 때 IK 가
몸통을 돌려 버리는 허우적거림(dance2 129초 IGRIS, run2 199·205초 K1)이 여기에 잡힌다. G1 은 두 수 모두 0 에 가깝다.
발·골반 높이·토크만 보던 이전 검사(ref_variant_eval.py)는 이것을 놓쳤다.
"""
import pickle, sys, glob, os, json
import numpy as np, mujoco
from general_motion_retargeting import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from scipy.spatial.transform import Rotation as R

FPS, THR = 30, 300.0
TRUNK = {"pelvis", "torso", "torso_link"}


def ang_speed(quats_wxyz):
    r = R.from_quat(quats_wxyz, scalar_first=True)
    return np.degrees((r[1:] * r[:-1].inv()).magnitude()) * FPS


def pairs(robot):
    d = json.load(open(IK_CONFIG_DICT["bvh_lafan1"][robot]))
    return [(v[0].replace("FootMod", "Foot"), link) for link, v in d["ik_match_table2"].items() if v[2] != 0]


def robot_speeds(robot, pkl, links):
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot])); d = mujoco.MjData(m)
    r = pickle.load(open(pkl, "rb"))
    q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l) for l in links]
    xq = np.zeros((len(q), len(links), 4))
    for t in range(len(q)):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(m, d); xq[t] = d.xquat[bid]
    return np.stack([ang_speed(xq[:, i]) for i in range(len(links))], 1)


def report(name, human, robot_sp, links):
    n = min(len(human), len(robot_sp)); h, rb = human[:n], robot_sp[:n]
    excess = rb - h; trunk = [i for i, l in enumerate(links) if l in TRUNK]
    bad_any = (excess > THR).any(1); bad_trunk = (excess[:, trunk] > THR).any(1)
    worst = np.unravel_index(np.argmax(excess[:, trunk]), excess[:, trunk].shape)
    print(f"  {name:18s} 몸통 허우적 {bad_trunk.sum():4d}프레임 ({bad_trunk.sum() / FPS:5.1f}s)  전체 {bad_any.sum():4d}프레임"
          f"  몸통 최악 {links[trunk[worst[1]]]} {worst[0] / FPS:.2f}s 로봇 {rb[worst[0], trunk[worst[1]]]:.0f} vs 사람 {h[worst[0], trunk[worst[1]]]:.0f}°/s")
    return bad_trunk.sum()


dirs = [a.split(":", 1) for a in sys.argv[1:]]
seqs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{dirs[-1][1]}/*.pkl"))
for seq in seqs:
    fr, _ = load_bvh_file(f"/home/sehoon/data/lafan1/{seq}.bvh", format="lafan1")
    print(f"== {seq}")
    for robot, dd in [("unitree_g1", "outputs/retarget")] + dirs:
        p = f"{dd}/{seq}.pkl"
        if not os.path.exists(p):
            continue
        pr = pairs(robot); links = [l for _, l in pr]
        human = np.stack([ang_speed(np.array([f[b][1] for f in fr])) for b, _ in pr], 1)
        report(os.path.basename(dd.rstrip("/")) if dd != "outputs/retarget" else "G1", human, robot_speeds(robot, p, links), links)
