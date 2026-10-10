"""같은 LAFAN1 클립을 G1 · 사람(bvh) · IGRIS-C · ROBOTIS K1 으로 나란히 그린다(GMR 리타게팅, 운동학만).

    MUJOCO_GL=egl python scripts/igris/retarget_4panel.py --seq walk1_subject2 --seconds 60 --out /tmp/x.mp4

패널마다 자기 모델을 따로 렌더하고 가로로 붙인다. 프레임마다 루트 xy 를 빼서 넷이 제자리에서 움직이게 한다
(GMR 은 루트 궤적도 스케일하므로 그냥 두면 서로 멀어진다). 자막·제목은 ffmpeg drawtext 로 뒤에 입힌다.
"""
import argparse, os, pickle, subprocess
import numpy as np, mujoco, imageio
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="walk1_subject2")
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--panel", type=int, nargs=2, default=(480, 1080))
ap.add_argument("--start", type=float, default=0, help="시작 초")
ap.add_argument("--panels", nargs="*", help="패널 지정 '로봇:pkl폴더:자막' (human 은 'human::자막'). 기본은 G1|사람|IGRIS|K1")
ap.add_argument("--title", default=None)
args = ap.parse_args()
W, H = args.panel
# (로봇, pkl, 자막, 보이는 geom group) — 충돌 메시는 숨긴다(G1·IGRIS 시각 메시는 group 1, K1 은 group 2)
PANELS = [("unitree_g1", "outputs/retarget/%s.pkl", "Unitree G1  1.32 m / 35 kg", 1),
          (None, None, "LAFAN1 human (bvh)", 0),
          ("igris_c", "outputs/retarget_igris/%s.pkl", "IGRIS-C  1.5 m / 58 kg", 1),
          ("robotis_k1", "outputs/retarget_k1/%s.pkl", "ROBOTIS AI Sapiens K1  1.2 m / 36 kg", 2)]
GRAY = np.array([0.42, 0.42, 0.44, 1.0], dtype=np.float32)
GROUP = {"unitree_g1": 1, "igris_c": 1, "robotis_k1": 2}
if args.panels:
    PANELS = []
    for spec in args.panels:
        robot, pkl, lab = spec.split(":", 2)
        PANELS.append((None, None, lab, 0) if robot == "human" else (robot, pkl + "/%s.pkl", lab, GROUP[robot]))
BONES = [("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"), ("Neck", "Head"),
         ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
         ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
         ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
         ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToe")]
ORANGE = np.array([0.95, 0.55, 0.1, 1.0], dtype=np.float32)

n = int(args.seconds * args.fps); s0 = int(args.start * args.fps)
frames, _ = load_bvh_file(f"/home/sehoon/data/lafan1/{args.seq}.bvh", format="lafan1")
frames = frames[s0:s0 + n]


def camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0, 0, 0.8); cam.distance = 3.4; cam.elevation = -10; cam.azimuth = 135
    return cam


def uniform(m, group):
    """모델마다 다른 하늘·조명·바닥을 끄고 같은 헤드라이트와 바닥 평면을 쓴다."""
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    m.vis.headlight.ambient[:] = 0.45; m.vis.headlight.diffuse[:] = 0.55; m.vis.headlight.specular[:] = 0.1
    if m.nlight:
        m.light_active[:] = 0
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[group] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
    return opt


def floor(scene):
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_PLANE, np.array([8.0, 8.0, 0.02]), np.zeros(3), np.eye(3).flatten(), GRAY)
        scene.ngeom += 1


class RobotPanel:
    def __init__(self, robot, pkl, group):
        self.m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot])); self.d = mujoco.MjData(self.m)
        self.opt = uniform(self.m, group)
        r = pickle.load(open(pkl, "rb"))
        self.q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)[s0:s0 + n]
        self.ren = mujoco.Renderer(self.m, height=H, width=W); self.cam = camera()

    def frame(self, t):
        q = self.q[min(t, len(self.q) - 1)].copy(); q[:2] = 0
        self.d.qpos[:] = q; mujoco.mj_forward(self.m, self.d)
        self.ren.update_scene(self.d, camera=self.cam, scene_option=self.opt); floor(self.ren.scene)
        self.ren.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        return self.ren.render()


class HumanPanel:
    """빈 바닥 모델 위에 뼈대를 캡슐·구로 그린다."""
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
        self.opt = uniform(self.m, 0)
        self.d = mujoco.MjData(self.m); self.ren = mujoco.Renderer(self.m, height=H, width=W); self.cam = camera()

    def frame(self, t):
        f = frames[t]; hip = np.array(f["Hips"][0]); off = np.array([hip[0], hip[1], 0.0])
        P = {k: np.array(v[0]) - off for k, v in f.items()}
        mujoco.mj_forward(self.m, self.d); self.ren.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        sc = self.ren.scene; floor(sc); sc.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        for a, b in BONES:
            if a in P and b in P and sc.ngeom < sc.maxgeom:
                g = sc.geoms[sc.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), ORANGE)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.018, P[a], P[b]); sc.ngeom += 1
        for k, p in P.items():
            if sc.ngeom < sc.maxgeom:
                g = sc.geoms[sc.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0]), p, np.eye(3).flatten(), np.array([1, 0.85, 0.3, 1], dtype=np.float32)); sc.ngeom += 1
        return self.ren.render()


panels = [HumanPanel() if r is None else RobotPanel(r, pkl % args.seq, grp) for r, pkl, _, grp in PANELS]
raw = args.out + ".raw.mp4"
w = imageio.get_writer(raw, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
for t in range(n):
    w.append_data(np.concatenate([p.frame(t) for p in panels], axis=1))
w.close()

B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
esc = lambda t: t.replace("\\", "\\\\").replace("%", "%%").replace(":", "\\:").replace("'", "\\'")  # drawtext 의 특수문자(%, :, ')
title = esc(args.title or f"Same LAFAN1 clip, three humanoids  ·  {args.seq}  ·  GMR retargeting")
labels = ",".join(f"drawtext=fontfile={R}:text='{esc(lab)}':fontsize=22:fontcolor={'0xf0a050' if r is None else 'white'}:x={i * W}+({W}-tw)/2:y=h-50" for i, (r, _, lab, _) in enumerate(PANELS))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf",
                f"drawbox=x=0:y=0:w=iw:h=56:color=black@0.85:t=fill,drawbox=x=0:y=ih-70:w=iw:h=70:color=black@0.85:t=fill,"
                f"drawtext=fontfile={B}:text='{title}':fontsize=30:fontcolor=white:x=24:y=13,"
                f"drawtext=fontfile={R}:text='%{{eif\\:t+{args.start}\\:d}}.%{{eif\\:mod((t+{args.start})*10\\,10)\\:d}} s':fontsize=24:fontcolor=0xc8c8c8:x=w-tw-24:y=16,{labels}",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", args.out], check=True)
os.remove(raw); print(args.out)
