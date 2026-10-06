"""python scripts/igris/jump_compare.py  (GMR 환경, 저장소 루트에서)
dance2 턱점프(130-142초) — 사람 원본 vs G1 레퍼런스 vs IGRIS v3 레퍼런스.
골반 높이·발 최저점·무게중심을 각자 서 있는 높이로 나눠 비교하고, 공중 구간(두 발 다 떨어짐)의 시작·끝을 본다."""
import pickle, numpy as np, mujoco
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

FPS, A, B = 30, 130, 142
SEQ = "dance2_subject3"


def summary(name, root_z, foot_z, com_z):
    t = np.arange(len(root_z)) / FPS
    g = np.percentile(foot_z, 5)          # 바닥 높이 추정(클립 전체 발 최저점 5 백분위)
    stand = np.median(root_z)             # 서 있는 골반 높이
    w = (t >= A) & (t < B)
    air = w & (foot_z > g + 0.03)
    seg = np.flatnonzero(air)
    rows = []
    if len(seg):
        cuts = np.split(seg, np.flatnonzero(np.diff(seg) > 1) + 1)
        for c in cuts:
            if len(c) < 3:
                continue
            rows.append(f"{t[c[0]]:.2f}-{t[c[-1]]:.2f}s ({len(c) / FPS:.2f}s) 발최고 {(foot_z[c].max() - g) * 100:.0f}cm")
    i = np.flatnonzero(w)
    lo, hi = root_z[i].min(), root_z[i].max()
    print(f"{name:8s} 서있는 골반 {stand:.2f}m | 창 안 골반 최저 {lo / stand:.2f} 최고 {hi / stand:.2f} (×서있는 높이)"
          f" | 무게중심 상승 {(com_z[i].max() - com_z[i].min()) * 100:.0f}cm ({(com_z[i].max() - com_z[i].min()) / stand:.2f}×)"
          f" | 공중: {'; '.join(rows) or '없음'}")


# 사람
fr, _ = load_bvh_file(f"/home/sehoon/data/lafan1/{SEQ}.bvh", format="lafan1")
hip = np.array([f["Hips"][0][2] for f in fr])
foot = np.array([min(f[k][0][2] for k in ("LeftFoot", "RightFoot", "LeftToe", "RightToe")) for f in fr])
com = np.array([np.mean([f[k][0][2] for k in f]) for f in fr])  # ponytail: 관절 평균으로 대신한 무게중심, 질량 모델 없음
summary("human", hip, foot, com)

for robot, pkl in (("unitree_g1", "outputs/retarget"), ("igris_c", "outputs/retarget_igris")):
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot])); d = mujoco.MjData(m)
    r = pickle.load(open(f"{pkl}/{SEQ}.pkl", "rb"))
    q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    feet = [g for g in range(m.ngeom) if m.geom_contype[g] and any(s in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for s in ("ankle_roll", "foot"))]
    fz, cz = [], []
    for k in range(len(q)):
        d.qpos[:] = q[k]; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        fz.append(min(d.geom_xpos[g][2] - (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][0]) for g in feet))
        cz.append(d.subtree_com[1][2])
    summary(robot.split("_")[0], q[:, 2], np.array(fz), np.array(cz))
