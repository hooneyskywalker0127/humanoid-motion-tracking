"""리타게팅 레퍼런스의 팔 방향 오차: 위팔(어깨→팔꿈치)·아래팔(팔꿈치→손목) 방향을 사람 BVH 와 비교한 각도(°).

    python scripts/igris/ref_arm_error.py robot:pkl폴더 [...]   (GMR 환경, 저장소 루트에서)

스케일과 무관한 방향만 본다. 팔 0자세 prior(--posture_zero) 같은 정규화가 팔을 사람에게서 얼마나 떼어 놓는지 잰다.
"""
import pickle, sys, glob, os
import numpy as np, mujoco
from general_motion_retargeting import ROBOT_XML_DICT

# (어깨, 팔꿈치, 손목) 바디 — 사람 쪽 (Arm, ForeArm, Hand)
ARMS = {"unitree_g1": (("left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link"), ("right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")),
        "robotis_k1": (("left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_rubber_hand"), ("right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_rubber_hand")),
        "igris_c": (("l_upper_arm", "l_elbow_bracket", "l_wrist_connector"), ("r_upper_arm", "r_elbow_bracket", "r_wrist_connector"))}
HUMAN = (("LeftArm", "LeftForeArm", "LeftHand"), ("RightArm", "RightForeArm", "RightHand"))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def ang(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def robot_dirs(robot, pkl):
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot])); d = mujoco.MjData(m)
    r = pickle.load(open(pkl, "rb"))
    q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    ids = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in arm] for arm in ARMS[robot]]
    out = np.zeros((len(q), 2, 2, 3))
    for t in range(len(q)):
        d.qpos[:] = q[t]; mujoco.mj_kinematics(m, d)
        for a, (s, e, w) in enumerate(ids):
            out[t, a, 0] = d.xpos[e] - d.xpos[s]; out[t, a, 1] = d.xpos[w] - d.xpos[e]
    return out


def human_dirs(frames):
    out = np.zeros((len(frames), 2, 2, 3))
    for t, f in enumerate(frames):
        for a, (s, e, w) in enumerate(HUMAN):
            out[t, a, 0] = np.array(f[e][0]) - np.array(f[s][0]); out[t, a, 1] = np.array(f[w][0]) - np.array(f[e][0])
    return out


if __name__ == "__main__":
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    dirs = [a.split(":", 1) for a in sys.argv[1:]]
    seqs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{dirs[-1][1]}/*.pkl"))
    for seq in seqs:
        fr, _ = load_bvh_file(f"/home/sehoon/data/lafan1/{seq}.bvh", format="lafan1"); h = human_dirs(fr)
        print(f"== {seq}")
        for robot, dd in [("unitree_g1", "outputs/retarget")] + dirs:
            p = f"{dd}/{seq}.pkl"
            if not os.path.exists(p):
                continue
            rb = robot_dirs(robot, p); n = min(len(rb), len(h)); e = ang(rb[:n], h[:n])
            up, lo = e[:, :, 0], e[:, :, 1]
            print(f"  {os.path.basename(dd.rstrip('/')) if dd != 'outputs/retarget' else 'G1':18s} 위팔 평균 {up.mean():4.1f}° p95 {np.percentile(up, 95):5.1f}°  아래팔 평균 {lo.mean():4.1f}° p95 {np.percentile(lo, 95):5.1f}°  >30° 프레임 {(e.max((1, 2)) > 30).mean() * 100:4.1f}%")
