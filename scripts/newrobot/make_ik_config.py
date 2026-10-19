"""새 휴머노이드의 GMR IK 설정(bvh_lafan1 → 로봇 json)을 MJCF 하나에서 만든다.

    python scripts/newrobot/make_ik_config.py <robot.xml> --out <json> --knee_torque <Nm> [--map role=body ...] [--trunk_rot 100] [--ground 0.03]

G1 설정(GMR bvh_lafan1_to_g1.json)을 기준으로:
  1. 링크 매핑 초안: 이름 규칙으로 pelvis·thigh·shank·foot·torso·upper_arm·forearm·hand 를 찾는다(미탐지는 오류, --map 으로 지정).
  2. 스케일: 영자세 FK 로 잰 다리(골반→발바닥)·팔(어깨→팔꿈치→손 구간 합)·몸통(골반→어깨) 길이를 G1 길이/G1 스케일
     (0.9/0.75/0.9)로 역산한 사람 길이로 나눈다. 점프 능력 지수 τ_knee/(m·g·L)(--knee_torque, 벤더 값) 가 G1(0.52) 보다
     작은 로봇은 다리·골반 스케일을 G1 값(0.9)으로 묶는다 — IGRIS-C(0.29)에서 1.01 은 0%, 0.90 은 100% 였다(261018).
     검증(261019): IGRIS-C 자동 설정으로 dance2 를 다시 리타게팅하면 손으로 만든 v4 와 표가 같고 채점 수치도 같다
     (체공 4 vs 5, 골반 +0.21 vs +0.22, 팔 오차 3.5°/11.6° vs 3.5°/11.5°); K1 은 표 차이 0. 팔·몸통 스케일만 실측값으로
     바뀐다(IGRIS 팔 0.97·몸통 1.10, K1 팔 0.53·몸통 0.95 — 손 설정은 G1 값 0.75/0.9 를 그대로 둔 것).
  3. 회전 오프셋: G1 오프셋을 복사하되, 영자세에서 몸체 좌표계로 본 다음 관절 방향이 G1 과 다르면(IGRIS 팔뚝은 -z, G1 은 +x)
     그 차이만큼 돌린 오프셋을 쓴다(scripts/igris/make_ik_config.py 가 손으로 하던 Ry(-90°)).
  4. 가중치: G1 표 그대로 + pelvis·torso 회전 가중치(--trunk_rot). ground_height 는 --ground(기본 0.03).
출력 json 과 함께 retarget_all.py 에 줄 옵션 한 줄을 찍는다. 뒤이어 scripts/newrobot/scorecard.py 로 G1 과 비교해 통과해야 학습한다.
"""
import argparse, json, re
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R

G1_JSON = "/home/sehoon/Projects/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json"
G1_XML = "/home/sehoon/Projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
# G1 표의 로봇 몸체 → 역할. 왼쪽/오른쪽은 같은 규칙.
G1_ROLE = {"pelvis": "pelvis", "torso_link": "torso",
           "left_hip_yaw_link": "l_thigh", "left_knee_link": "l_shank", "left_ankle_roll_link": "l_foot",
           "right_hip_yaw_link": "r_thigh", "right_knee_link": "r_shank", "right_ankle_roll_link": "r_foot",
           "left_shoulder_yaw_link": "l_upper_arm", "left_elbow_link": "l_forearm", "left_wrist_yaw_link": "l_hand",
           "right_shoulder_yaw_link": "r_upper_arm", "right_elbow_link": "r_forearm", "right_wrist_yaw_link": "r_hand"}
# 역할별 이름 규칙(우선순위 순). side 는 (left|l) 또는 (right|r).
RULES = {"pelvis": [r"^pelvis$", r"^base$", r"pelvis"], "torso": [r"^torso(_link)?$", r"torso", r"waist_yaw_link", r"^body$"],
         "thigh": [r"{s}_?hip_yaw", r"{s}_?upper_leg", r"{s}_?thigh", r"{s}_?hip_pitch"],
         "shank": [r"{s}_?knee", r"{s}_?lower_leg", r"{s}_?shank", r"{s}_?calf"],
         "foot": [r"{s}_?foot_original", r"{s}_?ankle_roll", r"{s}_?foot$", r"{s}_?foot", r"{s}_?ankle"],
         "upper_arm": [r"{s}_?shoulder_yaw", r"{s}_?upper_arm", r"{s}_?shoulder_roll"],
         "forearm": [r"{s}_?elbow", r"{s}_?forearm", r"{s}_?lower_arm"],
         "hand": [r"{s}_?wrist_yaw", r"{s}_?wrist_connector", r"{s}_?wrist_roll_rubber_hand", r"{s}_?hand$", r"{s}_?wrist_roll", r"{s}_?wrist", r"{s}_?hand"],
         "shoulder": [r"{s}_?shoulder_pitch", r"{s}_?shoulder"]}  # 길이 측정용: 어깨 사슬의 첫 몸체
CHILD = {"thigh": "shank", "shank": "foot", "upper_arm": "forearm", "forearm": "hand"}  # 방향 보정용 다음 관절
LEN_HUMAN = None  # G1 길이 / G1 스케일(0.9, 0.75, 0.9) 로 아래에서 정한다 = 1.8 m 사람 기준 길이
G1_SCALE = {"leg": 0.9, "arm": 0.75, "torso": 0.9}
KNEE_TORQUE = {"g1": 139.0}  # G1 무릎 토크(Nm). 새 로봇은 --knee_torque 로 준다(MJCF 의 ctrlrange 가 벤더 값이 아닐 수 있어 자동으로 읽지 않는다)


def body_names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]


def find(names, role, side=None):
    pats = RULES[role.split("_", 1)[1] if side else role]
    s = {"l": r"(left|l)", "r": r"(right|r)"}.get(side, "")
    for p in pats:
        hits = [n for n in names if re.search(p.format(s=s), n.lower())]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def mapping(m, overrides):
    names = body_names(m); out = {}
    for role in ("pelvis", "torso"):
        out[role] = find(names, role)
    for side in "lr":
        for part in ("thigh", "shank", "foot", "upper_arm", "forearm", "hand", "shoulder"):
            out[f"{side}_{part}"] = find(names, f"{side}_{part}", side)
    out.update(overrides)
    missing = [k for k, v in out.items() if v is None]
    assert not missing, f"몸체를 못 찾았다: {missing} (--map role=body 로 지정)"
    return out


def zero_pose_fk(m):
    d = mujoco.MjData(m); mujoco.mj_kinematics(m, d); return d


def seg_dir_local(m, d, body, child):
    """몸체 좌표계에서 본 (child 원점 − body 원점) 단위 벡터"""
    b, c = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, x) for x in (body, child))
    v = d.xmat[b].reshape(3, 3).T @ (d.xpos[c] - d.xpos[b])
    return v / np.linalg.norm(v)


def sole_z(m, d, foot):
    """발 몸체(와 그 아래)의 충돌 형상 최저 z"""
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, foot); zs = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b > 0 and b != fid:
            b = m.body_parentid[b]
        if b == fid and m.geom_contype[g]:
            sz = m.geom_size[g]; r = sz[2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else sz[0]
            zs.append(d.geom_xpos[g][2] - r)
    return min(zs) if zs else d.xpos[fid][2]


def lengths(m, d, mp):
    p = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
    leg = p(mp["pelvis"])[2] - np.mean([sole_z(m, d, mp[f"{s}_foot"]) for s in "lr"])
    # 팔은 구간 길이의 합(G1 영자세는 팔꿈치가 굽어 있어 직선 거리가 아니다)
    arm = np.mean([np.linalg.norm(p(mp[f"{s}_forearm"]) - p(mp[f"{s}_shoulder"])) + np.linalg.norm(p(mp[f"{s}_hand"]) - p(mp[f"{s}_forearm"])) for s in "lr"])
    torso = np.mean([np.linalg.norm(p(mp[f"{s}_shoulder"]) - p(mp["pelvis"])) for s in "lr"])
    return {"leg": leg, "arm": arm, "torso": torso}


def jump_index(m, mp, tau):
    """무릎 토크 / (질량·g·다리길이): G1 0.51, IGRIS-C 0.30, K1 0.34 — 점프 능력의 거친 지표"""
    if not tau:
        return None
    mass = float(m.body_subtreemass[1]); d = zero_pose_fk(m)
    return tau / (mass * 9.81 * lengths(m, d, mp)["leg"])


ap = argparse.ArgumentParser()
ap.add_argument("xml"); ap.add_argument("--out", required=True)
ap.add_argument("--map", nargs="*", default=[], help="역할=몸체 (예: l_foot=l_foot_original)")
ap.add_argument("--trunk_rot", type=float, default=100, help="pelvis·torso 회전 가중치(두 표). IGRIS 200, K1 100 으로 학습 성공")
ap.add_argument("--ground", type=float, default=0.03)
ap.add_argument("--no_cap", action="store_true", help="점프 능력 지수에 따른 다리 스케일 제한을 끈다")
ap.add_argument("--knee_torque", type=float, default=None, help="무릎 토크 한계(Nm, 벤더 값). 없으면 점프 능력 제한을 못 건다")
args = ap.parse_args()

mr = mujoco.MjModel.from_xml_path(args.xml); dr = zero_pose_fk(mr)
mg = mujoco.MjModel.from_xml_path(G1_XML); dg = zero_pose_fk(mg)
mp = mapping(mr, dict(kv.split("=") for kv in args.map))
g1map = {v: k for k, v in G1_ROLE.items()}  # 역할 → G1 몸체
g1map.update({"l_shoulder": "left_shoulder_pitch_link", "r_shoulder": "right_shoulder_pitch_link"})
L = lengths(mr, dr, mp); Lg = lengths(mg, dg, g1map)
LEN_HUMAN = {k: Lg[k] / G1_SCALE[k] for k in Lg}
scale = {k: L[k] / LEN_HUMAN[k] for k in L}
ji, jg = jump_index(mr, mp, args.knee_torque), jump_index(mg, g1map, KNEE_TORQUE["g1"])
if ji is None:
    print("경고: --knee_torque 가 없어 점프 능력 제한을 못 건다(무거운 로봇은 다리 스케일 0.9 권장)")
capped = False
if not args.no_cap and ji is not None and jg is not None and ji < jg and scale["leg"] > 0.9:
    scale["leg"], capped = 0.9, True

cfg = json.load(open(G1_JSON))
cfg["robot_root_name"] = mp["pelvis"]; cfg["ground_height"] = args.ground
for t in ("ik_match_table1", "ik_match_table2"):
    new = {}
    for g1body, entry in cfg[t].items():
        role = G1_ROLE[g1body]; body = mp[role]; e = list(entry)
        part = role.split("_", 1)[1] if "_" in role else role
        if part in CHILD:  # 다음 관절 방향이 G1 과 다르면 그만큼 돌린 오프셋
            side = role[0]; child_role = f"{side}_{CHILD[part]}"
            vg = seg_dir_local(mg, dg, g1body, g1map[child_role]); vr = seg_dir_local(mr, dr, body, mp[child_role])
            fix, _ = R.align_vectors([vg], [vr])  # 로봇 방향 → G1 방향 (IGRIS 팔뚝 -z → +x 가 Ry(-90°))
            fix = R.from_matrix(np.round(fix.as_matrix()))  # 축 관습 차이는 90° 단위라 반올림(G1 팔뚝이 x 축에서 3° 벗어난 것을 따라가지 않게)
            if np.degrees(np.arccos(np.clip(vg @ vr, -1, 1))) > 45:  # 관절 원점 어긋남(G1 고관절 ~20°)은 무시, 축 관습 차이(90°)만
                e[4] = [round(float(x), 8) for x in (R.from_quat(entry[4], scalar_first=True) * fix).as_quat(scalar_first=True)]
                cfg.setdefault("_notes", []).append(f"{body}: 다음 관절 방향 G1 {np.round(vg, 2).tolist()} → {np.round(vr, 2).tolist()}, 오프셋 보정")
        elif part == "hand":  # 끝 몸체는 팔뚝과 같은 보정
            fa_body = mp[f"{role[0]}_forearm"]
            if fa_body in new and new[fa_body][4] != cfg[t][g1map[f"{role[0]}_forearm"]][4]:
                fix = R.from_quat(cfg[t][g1map[f"{role[0]}_forearm"]][4], scalar_first=True).inv() * R.from_quat(new[fa_body][4], scalar_first=True)
                e[4] = [round(float(x), 8) for x in (R.from_quat(entry[4], scalar_first=True) * fix).as_quat(scalar_first=True)]
        if role in ("pelvis", "torso"):
            e[2] = args.trunk_rot
        new[body] = e
    cfg[t] = new
hs = cfg["human_scale_table"]
for k in hs:
    hs[k] = round(scale["leg"] if k in ("Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFootMod", "RightFootMod") else scale["torso"] if k == "Spine2" else scale["arm"], 3)
json.dump(cfg, open(args.out, "w"), indent=1)
print(f"매핑: {mp}")
print(f"길이(m): 다리 {L['leg']:.3f} 팔 {L['arm']:.3f} 몸통 {L['torso']:.3f}  (G1 {Lg['leg']:.3f}/{Lg['arm']:.3f}/{Lg['torso']:.3f})")
print(f"스케일: 다리 {scale['leg']:.2f}{' (점프 능력 지수 %.2f < G1 %.2f 라 0.90 으로 제한)' % (ji, jg) if capped else ''} 팔 {scale['arm']:.2f} 몸통 {scale['torso']:.2f} | 점프 능력 지수 {ji} (G1 {jg})")
for n in cfg.get("_notes", []):
    print("오프셋:", n)
print(f"→ {args.out}\nretarget_all.py 옵션: --ik_config {args.out} --posture 1.0 --contact_weight 100 20 --contact_threshold 0.08 --ground_lift --lift_window 1 --lift_sigma 0 --vel_limit shoulder 8 --vel_limit elbow 8 --vel_limit wrist 8")
