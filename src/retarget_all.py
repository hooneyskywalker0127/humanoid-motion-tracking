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
parser.add_argument("--ground_lift", action="store_true",
                    help="IK 뒤 발바닥(발 충돌 상자)이 바닥 아래면 그만큼 몸 전체를 올린다. 0.5초 창 최대값을 가우시안으로 펴 "
                         "튐을 막는다. 261004 IGRIS-C: 발 가중치를 올려 막던 관통을 이걸로 바꾸자 점프가 부풀지 않았다 "
                         "(dance2 136초 점프 0.59m → 0.49m, 사람 0.47m). OmniRetarget·KungfuBot 의 접지 보정과 같은 생각.")
args = parser.parse_args()


def ground_lift(model, qpos):
    """qpos(T, nq) 의 루트 z 를 발 충돌 상자가 바닥(z=0) 위에 오도록 올린 사본을 돌려준다."""
    import itertools
    import mujoco
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    data = mujoco.MjData(model)
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    boxes = [g for g in range(model.ngeom)
             if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_contype[g]
             and "foot" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])]
    assert boxes, "발 충돌 상자를 못 찾았다"
    sole = np.empty(len(qpos))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        sole[t] = min(((data.geom_xpos[g] + (signs * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T)[:, 2]).min()
                      for g in boxes)
    need = np.maximum(0.0, -sole)
    out = qpos.copy()
    out[:, 2] += np.maximum(need, gaussian_filter1d(maximum_filter1d(need, 15), 4))
    return out


class GMRPosture(GMR):
    """두 단계 IK 모두에 PostureTask 를 더한다. 목표는 그 프레임을 풀기 직전의 자세(= 직전 프레임 해)라,
    같은 손 위치를 내는 다른 IK 해로 갈아타는 것을 막는다. 루트(자유 관절)에는 걸리지 않는다."""

    def __init__(self, *a, posture, **k):
        super().__init__(*a, **k)
        import mink
        self.posture = mink.PostureTask(self.model, cost=posture)
        for tasks, errs in ((self.tasks1, self.task_errors1), (self.tasks2, self.task_errors2)):
            tasks.append(self.posture)
            errs[self.posture] = []

    def retarget(self, human_data, **k):
        self.posture.set_target(self.configuration.q.copy())
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
