"""학습 전 레퍼런스 검사. 리타게팅 결과(pkl)가 로봇이 따라갈 수 있는 동작인지 클립마다 잰다.

    python src/reference_qa.py --robot igris_c --pkl outputs/retarget_igris --out outputs/metrics/ref_qa_igris.csv
    python src/reference_qa.py --robot unitree_g1 --pkl outputs/retarget --seq dance2_subject3 --window 130 140

운동학
  flip        한 프레임(1/30초)에 관절 속도 25 rad/s 넘는 프레임 수 (IK 해 뒤집힘)
  limit_hold  관절이 한계 0.02 rad 안에 붙어 있는 프레임 비율 (가장 나쁜 관절)
  pen         발 충돌 형상이 바닥 아래로 들어간 최대 깊이 (cm)
  skate       땅을 디딘 발(바닥 위 1 cm 안)의 수평 속도 95백분위 (m/s). 0 에 가까워야 한다
물리 (역동역학 + 접촉 힘 분배)
  접촉 없이 mj_inverse 로 필요한 일반화 힘을 구하고, 바닥 1.5cm 안의 발바닥 점들에서 낼 수 있는 힘(위로만,
  마찰계수 1 의 4면 피라미드)으로 루트 6축 힘을 최소제곱(NNLS)으로 설명한다. 그 접촉 힘을 뺀 나머지가 관절
  토크이고, 설명 못 한 루트 힘이 잔차다. MuJoCo 소프트 접촉을 그대로 역산하면 margin 안의 뜬 발에도 큰
  반발력을 만들어 틀린다(첫 시도: IGRIS 걷기 잔차가 체중의 346배). 관절 마찰 손실도 끈다.
  vel_over    관절 속도가 학습 설정의 속도 한계를 넘는 프레임 비율과 가장 큰 배율
  tau_over    필요한 관절 토크가 학습 설정의 토크 한계를 넘는 프레임 비율과 가장 큰 배율
  residual    접촉으로 설명되지 않는 루트 힘의 95백분위 / 체중. 크면 레퍼런스 자체가 물리적으로 불가능한 가속이다
  com_out     접촉 프레임 중 무게중심 xy 가 발바닥 지지 다각형 밖 2cm 이상인 비율(PHUMA·R1 Pro 식 정적 안정 검사), p95 는 그 거리(cm)

한계값은 정책이 실제로 쓰는 Isaac 학습 설정(robots/igris.py, g1.py)과 같다. IGRIS 속도 한계는 그 파일에서도
가정값이다(공개 URDF 는 자리값 100). 자기 충돌은 재지 않는다 — IGRIS 벤더 메시가 설계상 겹쳐 학습에서도 껐다.
"""
import argparse
import glob
import itertools
import os
import pickle
import re

import mujoco
import numpy as np
from general_motion_retargeting import ROBOT_XML_DICT
from scipy.signal import savgol_filter

LIMITS = {  # (관절 정규식, 토크 Nm, 속도 rad/s) — 학습 설정과 같은 값
    "igris_c": [(r".*_hip_pitch", 150, 20), (r".*_knee_pitch", 150, 20), (r".*_hip_roll", 120, 20),
                (r".*_hip_yaw", 60, 37), (r".*_ankle_.*", 90, 32), (r"waist_.*", 60, 37),
                (r".*_shoulder_.*|.*_elbow_pitch", 60, 37), (r".*_wrist_.*", 8, 22), (r"neck_.*", 7, 10)],
    "robotis_k1": [(r".*_(hip_.*|knee|ankle_pitch)_joint", 96.9, 20), (r".*_ankle_roll_joint", 47.3, 32), (r"waist_yaw_joint", 96.9, 32),
                   (r".*_(shoulder_.*|elbow|wrist_roll)_joint", 47.3, 37)],  # 공개 MJCF actuator ctrlrange, 속도는 G1 처럼 가정
    "unitree_g1": [(r".*_hip_yaw_joint", 88, 32), (r".*_hip_roll_joint", 139, 20), (r".*_hip_pitch_joint", 88, 32),
                   (r".*_knee_joint", 139, 20), (r".*_ankle_.*", 50, 37), (r"waist_(roll|pitch)_joint", 50, 37),
                   (r"waist_yaw_joint", 88, 32), (r".*_shoulder_.*|.*_elbow_joint", 25, 37),
                   (r".*_wrist_roll_joint", 25, 37), (r".*_wrist_(pitch|yaw)_joint", 5, 22)],
}
FEET = {"igris_c": ("l_foot_original", "r_foot_original"), "robotis_k1": ("left_ankle_roll_link", "right_ankle_roll_link"),
        "unitree_g1": ("left_ankle_roll_link", "right_ankle_roll_link")}

parser = argparse.ArgumentParser()
parser.add_argument("--robot", required=True, choices=list(LIMITS))
parser.add_argument("--pkl", required=True)
parser.add_argument("--seq", nargs="*")
parser.add_argument("--window", nargs=2, type=float, help="이 구간(초)만 잰다")
parser.add_argument("--out")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
model.opt.timestep = 0.002  # IGRIS MJCF 는 0.00048 — 소프트 접촉의 시간상수가 dt 에 묶여 역동역학이 흔들린다
data = mujoco.MjData(model)
fps = 30
mass = float(model.body_mass.sum())
jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
tau_lim = np.zeros(len(jnames)); vel_lim = np.zeros(len(jnames))
for k, n in enumerate(jnames):
    hit = next((t, v) for rx, t, v in LIMITS[args.robot] if re.fullmatch(rx, n))
    tau_lim[k], vel_lim[k] = hit
jrange = model.jnt_range[1:]
foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET[args.robot]]
foot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in foot_bodies and model.geom_contype[g]]
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
signs = np.array(list(itertools.product([-1, 1], repeat=3)))


def geom_low(g):
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        return (data.geom_xpos[g] + (signs * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min()
    return data.geom_xpos[g][2] - model.geom_size[g][0]  # 구·캡슐 반지름


MU = 1.0
EDGES = np.array([[MU, 0, 1], [-MU, 0, 1], [0, MU, 1], [0, -MU, 1]]) / np.sqrt(1 + MU ** 2)


GROUND = [0.0]  # 클립마다 추정한 바닥 높이(PHUMA 식: 발 최저점 분포의 5 백분위). G1 레퍼런스는 발이 평균 1.5cm 떠 있다


def sole_points():
    """바닥 2cm 안의 발바닥 점들(월드 좌표, 몸체 id)."""
    pts = []
    for g in foot_geoms:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            c = data.geom_xpos[g] + (signs * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T
            c = c[np.argsort(c[:, 2])[:4]]  # 아래 네 꼭짓점
        else:
            c = (data.geom_xpos[g] - np.array([0, 0, model.geom_size[g][0]]))[None]
        pts += [(pt, model.geom_bodyid[g]) for pt in c if pt[2] < GROUND[0] + 0.02]
    return pts


def contact_split(qfrc):
    """qfrc(nv) 를 발 접촉 힘으로 설명한다. (관절 토크, 루트 잔차 힘 N) 을 돌려준다."""
    from scipy.optimize import nnls
    pts = sole_points()
    if not pts:
        return qfrc[6:], np.linalg.norm(qfrc[:3])
    cols = []
    for pt, b in pts:
        jacp = np.zeros((3, model.nv)); mujoco.mj_jac(model, data, jacp, None, pt, b)
        cols += [jacp.T @ e for e in EDGES]
    A = np.stack(cols, 1)  # nv x k
    coef, _ = nnls(A[:6], qfrc[:6], maxiter=200)
    rest = qfrc - A @ coef
    return rest[6:], np.linalg.norm(rest[:3])


def hull_dist(pts, c):
    """점 c 가 pts 의 볼록껍질 밖이면 껍질까지 거리, 안이면 0."""
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(pts)
    except Exception:
        return float(np.min(np.linalg.norm(pts - c, axis=1)))
    d = h.equations[:, :2] @ c + h.equations[:, 2]  # 바깥이 양수
    return float(max(0.0, d.max()))


def qa(path):
    r = pickle.load(open(path, "rb"))
    q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    if args.window:
        a, b = (int(x * fps) for x in args.window); q = q[a:b]
    T = len(q)
    # 속도: 위치 차분(쿼터니언은 mj_differentiatePos), 그 뒤 가속도. 잡음을 savgol 로 편다.
    v = np.zeros((T, model.nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(model, v[t], 1.0 / fps, q[t], q[t + 1])
    v[-1] = v[-2]
    v = savgol_filter(v, 9, 3, axis=0)
    acc = savgol_filter(np.gradient(v, 1.0 / fps, axis=0), 9, 3, axis=0)
    jv = np.abs(np.diff(q[:, 7:], axis=0)) * fps
    lows0 = []
    for t in range(0, T, 3):
        data.qpos[:] = q[t]; mujoco.mj_kinematics(model, data); lows0.append(min(geom_low(g) for g in foot_geoms))
    GROUND[0] = float(np.percentile(lows0, 5))
    tau = np.zeros((T, model.nv - 6)); res = np.zeros(T); low = np.zeros(T); skate = []; com_out = []
    for t in range(T):
        data.qpos[:] = q[t]; data.qvel[:] = v[t]; data.qacc[:] = acc[t]
        mujoco.mj_inverse(model, data)
        tau[t], res[t] = contact_split(data.qfrc_inverse.copy())
        pts = [p[:2] for p, _ in sole_points()]
        if len(pts) >= 3:  # 접촉 프레임만: 무게중심 xy 가 지지 다각형 밖이면 그 거리(m)
            mujoco.mj_comPos(model, data); com_out.append(hull_dist(np.array(pts), data.subtree_com[1][:2]))
        lows = [geom_low(g) for g in foot_geoms]; low[t] = min(lows)
        for b in foot_bodies:
            gb = [g for g in foot_geoms if model.geom_bodyid[g] == b]
            if min(geom_low(g) for g in gb) < GROUND[0] + 0.01:
                vel6 = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
                skate.append(np.linalg.norm(vel6[3:5]))
    near = (np.minimum(np.abs(q[:, 7:] - jrange[:, 0]), np.abs(q[:, 7:] - jrange[:, 1])) < 0.02).mean(0)
    vr = np.abs(v[:, 6:]) / vel_lim; tr = np.abs(tau) / tau_lim
    worst = lambda ratio: jnames[int(np.argmax(ratio.max(0)))]
    grp = {}
    for gname, rx in (("waist", r"waist.*"), ("hip", r".*hip.*"), ("knee", r".*knee.*"), ("ankle", r".*ankle.*"),
                      ("arm", r".*(shoulder|elbow).*")):
        idx = [k for k, n in enumerate(jnames) if re.fullmatch(rx, n)]
        grp[f"{gname}_over"] = f"{(tr[:, idx] > 1).any(1).mean():.3f}"
    return dict(frames=T, flip=int((jv.max(1) > 25).sum()),
                limit_hold=f"{near.max():.3f}", limit_joint=jnames[int(near.argmax())],
                pen_cm=f"{max(0.0, -low.min()) * 100:.1f}", skate_p95=f"{np.percentile(skate, 95) if skate else 0:.2f}",
                vel_over=f"{(vr > 1).any(1).mean():.3f}", vel_max=f"{vr.max():.2f}", vel_joint=worst(vr),
                tau_over=f"{(tr > 1).any(1).mean():.3f}", tau_max=f"{tr.max():.2f}", tau_joint=worst(tr),
                tau_p99=f"{np.percentile(tr.max(1), 99):.2f}",
                residual_med=f"{np.median(res) / (mass * 9.81):.2f}", residual_p95=f"{np.percentile(res, 95) / (mass * 9.81):.2f}",
                ground_cm=f"{GROUND[0] * 100:.1f}",
                com_out=f"{np.mean(np.array(com_out) > 0.02) if com_out else float('nan'):.3f}",  # 접촉 프레임 중 무게중심이 지지 다각형 밖 2cm 이상
                com_out_p95=f"{np.percentile(com_out, 95) * 100 if com_out else float('nan'):.1f}", **grp)


paths = [os.path.join(args.pkl, f"{s}.pkl") for s in args.seq] if args.seq else sorted(glob.glob(f"{args.pkl}/*.pkl"))
rows = []
for p in paths:
    row = {"seq": os.path.basename(p)[:-4], **qa(p)}
    rows.append(row)
    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
if args.out:
    import csv
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
