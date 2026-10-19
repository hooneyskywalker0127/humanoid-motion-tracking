# Adding a new humanoid

[← README](../README.en.md) · [한국어](new-robot.md)

The order of steps for putting a humanoid that GMR does not support onto this pipeline. It bundles what it took to add
IGRIS-C (ROBROS) and the ROBOTIS AI Sapiens K1 by hand into two scripts and a checklist. The point is to **score the reference
against the G1 before training**, so that a retargeting mistake does not surface only after 18 hours of training, as it did
with dance2 (0% on both robots, 2026-10-17).

```
A. Auto config   make_ik_config.py  MJCF → GMR IK json (mapping, scales, offsets, weights)
B. Human check   read the printed mapping and the zero pose, enter vendor values (knee torque)
C. Scorecard     retarget_all.py on 4 clips → scorecard.py compares with the G1 and the human, pass/fail
D. Training      only passing references go to Isaac Lab (register the robot using the K1 files as the template)
```

## A. Generating the GMR IK config

```bash
# GMR environment, repo root
python scripts/newrobot/make_ik_config.py ~/Projects/GMR/assets/<robot>/<robot>.xml \
    --out configs/gmr/bvh_lafan1_to_<robot>_candidate.json --knee_torque <Nm> [--trunk_rot 100] [--map l_foot=<body> ...]
```

Starting from the G1 config (`bvh_lafan1_to_g1.json`), it fills in four things.

| Item | Method | Basis |
|---|---|---|
| Link mapping | Name rules find pelvis, torso, thigh, shank, foot, shoulder, upper arm, forearm and hand. A miss is an error; override with `--map role=body` | GMR's new-robot PRs copy the G1 json and rename links |
| Scales | Zero-pose FK measures leg (pelvis → sole), arm (shoulder → elbow → hand) and torso (pelvis → shoulder) lengths and divides them by the human lengths back-computed from G1 length / G1 scale (0.9/0.75/0.9) | The meaning of GMR's `human_scale_table`. On the G1 it returns 0.90/0.75/0.90 |
| Leg scale cap | If the jump index τ_knee/(m·g·L_leg) is below the G1's (0.52), leg and pelvis scales are capped at 0.9 | IGRIS-C (0.29): 1.01 → dance2 0%, 0.90 → 100%. A heavy robot must not be asked for human-sized jumps |
| Rotation offsets | G1 offsets are copied; if the direction to the next joint in the body frame differs from the G1 by more than 45°, the offset is rotated by that difference (rounded to 90° steps) | IGRIS-C forearm points −z, G1 +x → Ry(−90°). The value found by hand now comes out automatically |
| Weights | G1 table + pelvis position 100, pelvis/torso rotation `--trunk_rot` (default 100), `ground_height` 0.03 | With a low trunk weight the IK rotates the torso when an arm target is out of reach (flailing) |

Validation (2026-10-19): the auto config for IGRIS-C has the same IK table (weights and offsets) as the hand-made retargeting v4, and
re-retargeting dance2 with it gives the same scores (flights 4 vs 5, pelvis apex +0.21 vs +0.22 m, arm direction error 3.5°/11.6° vs
3.5°/11.5°). For the K1 the table diff is zero. Only the arm and torso scales change: the hand-made configs kept the G1 values (0.75/0.9),
the generator measures them (IGRIS-C 0.97/1.10, K1 0.53/0.95).

## B. What a person checks

1. In the printed mapping line, that the foot and hand bodies are right (the foot is the body with the collision box, the hand is the body GMR uses as the wrist).
2. `--knee_torque` comes from the vendor spec, not the MJCF `ctrlrange`. Some MJCFs leave it empty or use other values.
3. For any body with an "offset corrected" line, look at the zero pose once in the MuJoCo viewer. An elbow frozen straight means a wrong offset (IGRIS-C, 2026-10-02).
4. Trunk weight: IGRIS-C trained successfully with 200, the K1 with 100. Raise it if the scorecard reports flailing.

## C. Retargeting and scoring

```bash
python src/retarget_all.py --src <bvh dir> --dst outputs/retarget_<robot> --robot <robot> \
    --ik_config configs/gmr/bvh_lafan1_to_<robot>_candidate.json \
    --posture 1.0 --contact_weight 100 20 --contact_threshold 0.08 --ground_lift --lift_window 1 --lift_sigma 0 \
    --vel_limit shoulder 8 --vel_limit elbow 8 --vel_limit wrist 8          # the generator prints this line
python scripts/newrobot/scorecard.py <robot>:outputs/retarget_<robot>
```

One line per clip; "pass" when every item is within its threshold. The absolute criteria from the literature (PHUMA joint limits at
98%, NMR joint jumps of 0.5 rad, non-floating 1 cm) are violated by the G1 reference itself, which completes 100%, so the scorecard
measures **differences from the G1** (joint saturation, joint jumps, penetration, floating, trunk flailing, arm direction, root pitch)
and **differences from the human** (both-feet airborne ratio, pelvis apex). The thresholds were set so that the references that scored
0% on dance2 (IGRIS-C E31, K1 P10) fail and the ones that scored 100% (IGRIS-C v4, K1 v1) pass. Definitions and thresholds are at the
top of `scripts/newrobot/scorecard.py`.

| Clip | IGRIS-C E31 (0%) | IGRIS-C v4 (100%) | K1 P10 (0%) | K1 v1 (100%) |
|---|---|---|---|---|
| dance2 | fail: air +0.09, apex +0.07 | pass | fail: air +0.10, apex +0.08 | pass |
| obstacles3 | fail: air | pass + knee warning (81%) | pass + knee warning | pass + knee warning (94%) |
| jumps1 | fail: air, apex | pass | pass | pass |
| run2 | fail: air, pitch | **fail: air +0.13, pitch 5%** | fail: air | pass |

The knee warning (rom) appears when the human knee flexion exceeds the robot's knee range in more than 2% of frames. That section is
kinematically infeasible, so below 95% is expected and the number is footnoted rather than fixed by retargeting (IGRIS-C 131°: 4.1% →
81%, K1 135°: 3.9% → 94%). The IGRIS-C v4 failure on run2 is a **prediction** with no training result yet (the thresholds will be
revisited with the result of 2026-10-19).

Where to fix each failing item: arm, pitch → mapping/offsets (A); air, apex, float → scales (A, do not pass `--no_cap`); pen, float →
`--contact_weight`/`--ground_lift`; jump, flail → `--vel_limit`/`--posture`/`--trunk_rot`.

## D. Registering in Isaac Lab

Use the K1 files as the template (fork `whole_body_tracking`).

- `robots/k1.py`: robot cfg, actuator PD (G1 ratio: kp = G1 kp/torque ratio × new robot torque, kd = 0.0637·kp), `*_ACTION_SCALE`,
  the retargeting joint order. Rewrite `package://` URDF paths to absolute ones as `scripts/k1/prepare_urdf.py` does.
- `tasks/tracking/config/k1/`: `__init__.py` (task registration), `flat_env_cfg.py` (body names, end effectors, undesired-contact regex),
  `agents/rsl_rl_ppo_cfg.py`.
- `scripts/csv_to_npz.py --robot k1`: reference pkl → csv → npz.
- Training, evaluation and video are one line: `scripts/rsl_rl/server_loop.sh <robot> <seq> <ref> <csv dir>` (30,000 iterations, frame 0, 100 rollouts, DR off).

## Not done yet

- The scorecard thresholds come from two robots and four clips only. They will be revisited once run2 and jumps1 finish.
- The foot and hand rules are name-based; a robot with unusual names needs `--map`.
