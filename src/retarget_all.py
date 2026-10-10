"""LAFAN1 bvh 전체를 G1으로 리타게팅해 pkl로 저장한다.

GMR의 bvh_to_robot_dataset.py는 이 버전에서 없는 함수(load_lafan1_file)와
없는 키(src_human="bvh")를 참조해 동작하지 않는다. 검증된 단일 경로
(bvh_to_robot.py와 같은 호출)를 그대로 반복한다.
"""
import argparse
import glob
import os
import pickle
import time

import numpy as np
import mujoco
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from tqdm import tqdm

# 첫 프레임을 몇 번 반복해 풀지. 한 번 호출이 최대 10회 반복이므로 20번이면 200회다.
FIRST_FRAME_PASSES = 20

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="/home/sehoon/data/lafan1")
parser.add_argument("--dst", default="/home/sehoon/Documents/GitHub/"
                                     "g1-motion-tracking/outputs/retarget")
parser.add_argument("--robot", default="unitree_g1")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--override", action="store_true")
parser.add_argument("--posture", type=float, default=0.0,
                    help="IK 에 직전 프레임 자세로 끄는 연속성 항(mink.PostureTask)을 이 가중치로 붙인다. "
                         "261003 IGRIS-C: 어깨 pitch·yaw 가 한 프레임에 ~1.5rad 뒤집히는 프레임이 14클립 52개 → "
                         "1.0 에서 4개. 손 추종 오차는 +4mm. 20 이상은 추종이 무뎌진다. G1 은 0 으로 그대로.")
parser.add_argument("--posture_zero", type=float, default=0.0, help="팔 관절(shoulder/elbow/wrist)을 0 자세로 당기는 PostureTask 가중치(--posture 와 함께)")
parser.add_argument("--posture_zero_joints", default="shoulder,elbow,wrist", help="--posture_zero 를 걸 관절 이름 조각(쉼표)")
parser.add_argument("--ik_range", nargs=3, action="append", metavar=("JOINT", "LO", "HI"), default=[],
                    help="IK 에서만 쓰는 관절 범위(rad). 로봇 한계 안에서 더 좁혀, 어깨가 ±π 로 감기는 등가 해를 막는다")
parser.add_argument("--vel_limit", nargs=2, action="append", metavar=("JOINT_SUBSTR", "RAD_S"), default=[],
                    help="IK 관절 속도 한도(--posture 와 함께). 이름에 JOINT_SUBSTR 가 든 관절의 프레임 사이 이동을 RAD_S/fps 로 묶는다. 예: --vel_limit shoulder 12 --vel_limit waist 6")
parser.add_argument("--ground_lift", action="store_true",
                    help="IK 뒤 발바닥(발 충돌 상자)이 바닥 아래면 그만큼 몸 전체를 올린다. 0.5초 창 최대값을 가우시안으로 펴 "
                         "튐을 막는다. 261004 IGRIS-C: 발 가중치를 올려 막던 관통을 이걸로 바꾸자 점프가 부풀지 않았다 "
                         "(dance2 136초 점프 0.59m → 0.49m, 사람 0.47m). OmniRetarget·KungfuBot 의 접지 보정과 같은 생각.")
parser.add_argument("--contact_weight", type=float, nargs=2, metavar=("ON", "OFF"),
                    help="발 위치 가중치를 사람 발이 바닥 3cm 안이면 ON, 떠 있으면 OFF 로 프레임마다 바꾼다(--posture 와 함께). "
                         "261010 IGRIS-C: 늘 높이면(v2, E3) 공중에서 발을 끌어올릴 때 골반까지 딸려 올라가 점프가 부푼다.")
parser.add_argument("--lift_window", type=int, default=15, help="--ground_lift 의 최대값 창(프레임). 1 이면 안 씀")
parser.add_argument("--lift_sigma", type=float, default=4, help="--ground_lift 의 가우시안 sigma(프레임). 0 이면 안 씀. "
                    "261010: 15/4 는 착지 관통 보정을 ±0.25초로 번지게 해 이착지 앞뒤 발을 들어 올린다(jumps1 체공 1.10초)")
parser.add_argument("--contact_threshold", type=float, default=0.03, help="--contact_weight 의 접촉 판정 높이(m, 사람 발 바닥 기준)")
parser.add_argument("--ik_config", help="이 json 을 IK 설정으로 쓴다(GMR 의 등록 파일 대신). 스케일·가중치 실험용")
args = parser.parse_args()
if args.ik_config:  # motion_retarget 은 params 의 dict 객체를 그대로 쓰므로 안에서 바꾸면 된다
    from general_motion_retargeting import params as _p
    _p.IK_CONFIG_DICT["bvh_lafan1"][args.robot] = args.ik_config


def ground_lift(model, qpos):
    """qpos(T, nq) 의 루트 z 를 발 충돌 상자가 바닥(z=0) 위에 오도록 올린 사본을 돌려준다."""
    import itertools
    import mujoco
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    data = mujoco.MjData(model)
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    # 발 충돌 형상: IGRIS 는 *_foot_* 몸체의 상자, G1·K1 은 *_ankle_roll_link 의 상자·구
    feet = [g for g in range(model.ngeom) if model.geom_contype[g]
            and int(model.geom_type[g]) in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE))
            and any(k in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) for k in ("foot", "ankle_roll"))]
    assert feet, "발 충돌 형상을 못 찾았다"

    def low(g):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            return ((data.geom_xpos[g] + (signs * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T)[:, 2]).min()
        return data.geom_xpos[g][2] - model.geom_size[g][0]
    sole = np.empty(len(qpos))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        sole[t] = min(low(g) for g in feet)
    need = np.maximum(0.0, -sole)
    out = qpos.copy()
    w, sg = args.lift_window, args.lift_sigma
    smooth = gaussian_filter1d(maximum_filter1d(need, w) if w > 1 else need, sg) if sg > 0 else need
    out[:, 2] += np.maximum(need, smooth)
    return out


class GMRPosture(GMR):
    """두 단계 IK 모두에 PostureTask 를 더한다. 목표는 그 프레임을 풀기 직전의 자세(= 직전 프레임 해)라,
    같은 손 위치를 내는 다른 IK 해로 갈아타는 것을 막는다. 루트(자유 관절)에는 걸리지 않는다."""

    def __init__(self, *a, posture, **k):
        super().__init__(*a, **k)
        import mink
        for name, lo, hi in args.ik_range:
            self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] = (float(lo), float(hi))
        self.posture = mink.PostureTask(self.model, cost=posture)
        extra = [self.posture]
        if args.posture_zero > 0:  # 팔 관절만 0 자세로 약하게 당긴다. 어깨가 ±π 로 감긴 등가 해(IGRIS 어깨 roll 범위 3.3rad)에 갇혔다가 한 번에 풀리는 것을 막는다
            cost = np.zeros(self.model.nv)
            for j in range(self.model.njnt):
                if any(s in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for s in args.posture_zero_joints.split(",")):
                    cost[self.model.jnt_dofadr[j]] = args.posture_zero
            self.posture_zero = mink.PostureTask(self.model, cost=cost)
            self.posture_zero.set_target(np.concatenate([[0, 0, 0, 1, 0, 0, 0], np.zeros(self.model.nq - 7)]))
            extra.append(self.posture_zero)
        for tasks, errs in ((self.tasks1, self.task_errors1), (self.tasks2, self.task_errors2)):
            for t in extra:
                tasks.append(t)
                errs[t] = []
        if args.vel_limit:  # 프레임당 관절 이동을 RAD_S/fps 로 묶는다. GMR 의 use_velocity_limit 은 dt=model timestep 이라 반복 횟수에 따라 한도가 달라져 쓰지 않는다
            # GMR 은 ik_limits 를 mink.solve_ik 의 6번째 자리(safety_break)에 넘겨 mink 가 무시하고 기본 ConfigurationLimit 만 쓴다. limits 자리로 돌려준다
            _solve_ik = mink.solve_ik

            def solve_ik(configuration, tasks, dt, solver, damping=1e-12, safety_break=False, limits=None, **kw):
                if not isinstance(safety_break, bool):
                    limits, safety_break = safety_break, False
                return _solve_ik(configuration, tasks, dt, solver, damping, safety_break, limits, **kw)
            mink.solve_ik = solve_ik
            self.vel_box = mink.ConfigurationLimit(self.model)
            self.jnt_lo, self.jnt_hi = self.vel_box.lower.copy(), self.vel_box.upper.copy()
            self.vel_step = np.full(self.model.nq, np.inf)
            for j in range(self.model.njnt):
                for sub, rad_s in args.vel_limit:
                    if sub in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j):
                        self.vel_step[self.model.jnt_qposadr[j]] = float(rad_s) / args.fps
            self.ik_limits.append(self.vel_box)

    def retarget(self, human_data, **k):
        self.posture.set_target(self.configuration.q.copy())
        if args.vel_limit:
            q = self.configuration.q
            self.vel_box.lower = np.maximum(self.jnt_lo, q - self.vel_step)
            self.vel_box.upper = np.minimum(self.jnt_hi, q + self.vel_step)
        if args.contact_weight:  # 발이 땅에 있을 때만 발 위치 가중치를 올린다(OmniRetarget·KungfuBot 의 접촉 인식 가중치와 같은 생각)
            on, off = args.contact_weight
            for foot in ("LeftFootMod", "RightFootMod"):
                w = on if human_data[foot][0][2] < self.foot_ground + args.contact_threshold else off
                for table in (self.human_body_to_task1, self.human_body_to_task2):
                    if foot in table:
                        table[foot].set_position_cost(w)
        return super().retarget(human_data, **k)

os.makedirs(args.dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(args.src, "*.bvh")))
print(f"{len(files)} bvh files")

failed = []
for path in files:
    seq = os.path.basename(path).replace(".bvh", "")
    out = os.path.join(args.dst, f"{seq}.pkl")
    if os.path.exists(out) and not args.override:
        print(f"skip {seq}")
        continue

    t0 = time.time()
    try:
        frames, human_height = load_bvh_file(path, format="lafan1")
        if args.posture > 0:
            retargeter = GMRPosture(src_human="bvh_lafan1", tgt_robot=args.robot,
                                    actual_human_height=human_height, posture=args.posture)
            retargeter.foot_ground = float(np.percentile([min(f["LeftFootMod"][0][2], f["RightFootMod"][0][2]) for f in frames], 5))
        else:
            retargeter = GMR(src_human="bvh_lafan1", tgt_robot=args.robot,
                             actual_human_height=human_height)
        # GMR 의 IK 는 직전 프레임 해에서 warm start 한다. 첫 프레임만 그게 없어
        # max_iter=10 안에 수렴하지 못하고 궤적을 벗어난 자세가 나온다
        # (motion_retarget.py:186, "curr_error - next_error > 0.001" 로 조기 종료).
        # 260912 측정: CSV 20개 중 11개에서 첫 구간 변화가 이후 중앙값의 5배 초과,
        # 최대 0.78 rad. 그 결함이 레퍼런스 속도 23 rad/s 스파이크로 이어지고,
        # 프레임 0 에서 시작하는 표준 평가에서 17개 중 8개가 무너졌다.
        # 첫 프레임을 여러 번 풀어 수렴시킨 뒤 본 루프를 돈다.
        for _ in range(FIRST_FRAME_PASSES):
            retargeter.retarget(frames[0])
        qpos = np.array([retargeter.retarget(f)
                         for f in tqdm(frames, desc=seq, leave=False)])
        if args.ground_lift:
            qpos = ground_lift(retargeter.model, qpos)
    except Exception as e:
        print(f"FAILED {seq}: {e}")
        failed.append((seq, str(e)))
        continue

    with open(out, "wb") as f:
        pickle.dump({
            "fps": args.fps,
            "root_pos": qpos[:, :3],
            # wxyz -> xyzw, bvh_to_robot.py의 저장 형식과 맞춘다
            "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
            "dof_pos": qpos[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }, f)
    print(f"{seq}  {len(frames)} frames  {time.time() - t0:.1f}s")

print()
print(f"완료 {len(files) - len(failed)} / {len(files)}")
for seq, err in failed:
    print(f"  실패 {seq}: {err}")
