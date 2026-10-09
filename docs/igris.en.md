# A second robot — IGRIS-C

[← README](../README.en.md)

To check that the pipeline is not tied to the G1, a second humanoid is being added:
[IGRIS-C](https://github.com/robrosinc/igris_c_description_public) (1.5 m, 58 kg,
31 DoF). Retargeting is done so far.

![retarget_igris](retarget_igris.gif)

The same LAFAN1 clip retargeted onto the G1 (left) and IGRIS-C (right), in one scene with
the human skeleton in the middle. GMR also scales the root trajectory, so the G1 walks
0.88 times the human's path. To keep them side by side, each robot's hip is moved onto
the human's horizontal hip position every frame; poses are untouched.

Adding a robot to GMR takes four things: the robot model, which human bone drives which
link, a length ratio per body part, and a rotation offset per link. The IGRIS-C table was
derived from the G1 one, and two things were wrong.

- **Forearms.** At zero pose the G1 forearm points forward and the IGRIS-C forearm points
  down. With the G1 offsets the elbows sat on their joint limit in 80-88% of frames.
  Turning the elbow and wrist offsets by 90° fixed it.
- **Leg ratio.** The IGRIS-C sole is 7.1 cm below the ankle joint, twice the G1's
  3.5 cm. Scaling the legs by ankle height sank the feet into the floor in all 14 clips.
  Scaling by sole height cut the clips with more than 3 cm of penetration from 14 to 4.
  The remaining four have crouching or crawling.

![igris_fix_arm](igris_fix_arm.png)

![igris_fix_leg](igris_fix_leg.png)

Each fix undone on its own and the same clip retargeted again (first 15 s of each clip).
With the G1 forearm offsets, the elbows in `aiming1_subject1` sit on the straight-arm limit
in every frame and the aiming pose never appears. Measuring the legs to the ankle puts the
soles of `walk1_subject1` below the floor in every frame, 1.8 cm on average and 3.9 cm at
worst. A physics simulator cannot reproduce a foot below the floor, so however well the policy
learns, that much stays as tracking error.

The model has no license file, so no IGRIS-C mesh or XML is in this repository. The
scripts in `scripts/igris/` read a local clone and write locally.

| File | What it does |
| --- | --- |
| `scripts/igris/prepare_mjcf.py` | MJCF for retargeting; drops 29 backlash joints and 22 finger joints |
| `scripts/igris/prepare_urdf.py` | URDF for training; fills the placeholder torque limits from the MJCF actuators |
| `scripts/igris/make_ik_config.py` | derives the IGRIS-C IK table from the G1 one |
| `scripts/igris/compare_retarget.py` | compares both robots' retargets on foot penetration, joint limits and velocity spikes |
| `scripts/igris/results_table.py` | builds the per-clip IGRIS vs. G1 table in the section below |
| `scripts/igris/fit_slot_map.py` | maps IGRIS-C joints onto the G1 policy's 29 slots (below) |
| `src/render_retarget_two.py` | renders the clip above |

**The G1 policy does not run on IGRIS-C as is.** Its inputs and outputs are laid out
for the G1's 29 joints and it learned the G1's mass and motors. Prior work,
[Any2Any](https://arxiv.org/abs/2605.23733), maps the joints onto the original
policy's slots and fine-tunes part of it on the new robot, beating training from
scratch at a fraction of the compute. The same comparison is set up here. Mapping by
joint name is wrong: the waist axes have opposite signs, the elbows zero 90° apart, and
because the forearms point along different axes the wrist roll and yaw swap. Signs and
offsets were fitted on the same 14 clips retargeted onto both robots; every one of the
29 pairs correlates at |r| ≥ 0.72.

## Training from scratch vs. transferring the G1 policy

![igris_transfer](igris_transfer.png)

The same clip was learned on IGRIS-C two ways. The clip is the most dynamic of the fourteen,
`run2_subject4` (running, 2.0 m/s root speed, 245 s).

- **A (from scratch):** BeyondMimic PPO on IGRIS-C from a random policy.
- **B (transfer):** the G1 teacher for the same clip (PPO), moved the
  [Any2Any](https://arxiv.org/abs/2605.23733) way. Joints are mapped onto the G1 slots (no training), the
  teacher's weights are frozen, and LoRA on every actor layer and the critic hidden layers is the only thing
  trained (rank 9, 5.0% of parameters, paper 5.26%). PPO settings are the teacher's own.

Both 30,000 iterations. Evaluation: 100 rollouts, domain randomization off, final checkpoint.

| Start frame | A success rate | B success rate | A posture error E_mpbpe | B posture error E_mpbpe |
|---|---|---|---|---|
| 0 | 21% | 0% | 59 mm | — |
| 1000 | 96% | 16% | 59 mm | 69 mm |
| 4000 | 98% | 35% | 59 mm | 71 mm |
| 8000 | 98% | 57% | 60 mm | 71 mm |

**On this setup, training from scratch wins.** B learns far faster at first: training episode length at
iteration 100 is 12.5 for A and 172 for B, and B passes in 100 iterations what A reaches at 1,000. Then it
stops lower. A frozen 35 kg G1 policy with a 5% low-rank correction does not seem to cover a 58 kg body with
different leg length and motors; I have not verified this. The paper transferred a large policy trained on
all of AMASS (SONIC); here the source is a single-clip teacher. The starting policies differ in scale.

A also manages only 21% from frame 0. Most rollouts fall in the first 20 seconds, going from standing into
a run; skip 1000 frames and it is 96-98%. Global position error is about 1.7 m for both, accumulated drift
over 245 s of running.

The comparison is per iteration. The ~9 hours spent training the G1 teacher that B starts from are not counted.

### Getting IGRIS-C to learn at all

The first two attempts did not learn: episode length 5 after 8,500 iterations (G1: 397 at the same point).

1. **PD gains.** The G1 rule (kp = armature·ω²) gave IGRIS a hip kp of 790 because of its large armature,
   shrinking one action unit to 0.047 rad (G1: 0.55). Switching to G1's kp/torque ratio alone did not help.
2. **Self-collision (the real cause).** Logging physics right after reset showed 31,000 N on the pelvis. The
   vendor meshes overlap by design at pelvis/thigh, torso/upper arm and hand/forearm, and PhysX was resolving
   collisions inside the body. With self-collision off, as in Isaac Lab's own G1 and H1 configs, the force
   went to 0 N and learning followed a G1-like curve.

### Retargeting v2 — stopping flips and penetration inside the IK

dance2 (0%) and obstacles3 (stalled training) were both reference problems, so I looked at how
retargeting is done "properly". The GMR paper itself lists one-frame joint jumps as a known artifact
and suggests per-motion weight tuning. Work that moved GMR to another robot
([wheeled humanoid](https://arxiv.org/abs/2609.11357)) added damping and velocity limits to the mink IK;
[PHUMA](https://arxiv.org/abs/2510.26236) adds grounding and anti-skating losses to the IK, and
[OmniRetarget](https://arxiv.org/abs/2509.26633) corrects the height to the floor every frame. The common
thread is adding **continuity** and **ground** constraints to GMR's frame-by-frame IK.

Two additions, without touching GMR (`src/retarget_all.py --posture`, `scripts/igris/make_ik_config.py`):

- **Continuity term.** A mink `PostureTask` on both IK stages, targeting the previous frame's solution
  (weight 1). Switching to another solution with the same hand position now costs something, and the
  shoulder flips go away. Above 20 the tracking goes soft.
- **Foot position weight 50 → 200.** The IGRIS-C knee only folds to 131° (G1: 165°), so in deep crouches
  the IK followed the pelvis target and pushed the feet under the floor. With heavier feet the pelvis stays
  higher and the feet stay on the ground: the obstacles3 crouch's lowest pelvis goes from 0.34 m to 0.50 m.
  The robot really cannot crouch that low, so that is the honest answer.

Measured over all 14 clips. Flips are frames with more than 25 rad/s in 1/30 s; hand error is the mean
distance to the scaled human wrist. GMR's velocity-limit option was unusable: it integrates with the
model timestep, and the IGRIS-C MJCF has 0.00048 s, which caps motion at 0.05 rad per frame.

| Clip | Flip frames v1 → v2 | Lowest sole (cm) v1 → v2 | Frames >3 cm under v1 → v2 | Hand error (mm) v1 → v2 |
|---|---|---|---|---|
| aiming1_subject1 | 0 → 0 | -4.8 → -3.4 | 1.9% → 0.1% | 99 → 101 |
| dance2_subject3 | 33 → 2 | -10.8 → -4.2 | 0.7% → 0.0% | 118 → 120 |
| jumps1_subject1 | 8 → 0 | -8.7 → -5.1 | 1.6% → 0.8% | 111 → 115 |
| obstacles3_subject3 | 1 → 0 | -19.4 → -3.0 | 4.6% → 0.0% | 106 → 115 |
| run2_subject4 | 2 → 1 | -4.5 → -2.2 | 0.3% → 0.0% | 124 → 129 |
| walk1_subject1 | 0 → 0 | -3.5 → -2.9 | 0.0% → 0.0% | 103 → 109 |
| walk1_subject2 | 0 → 0 | -2.2 → -2.7 | 0.0% → 0.0% | 107 → 112 |
| walk1_subject5 | 1 → 0 | -3.9 → -2.8 | 0.1% → 0.0% | 94 → 98 |
| walk2_subject1 | 4 → 1 | -7.0 → -4.7 | 1.5% → 0.7% | 120 → 124 |
| walk2_subject3 | 2 → 0 | -21.5 → -4.5 | 5.0% → 0.9% | 110 → 114 |
| walk2_subject4 | 0 → 0 | -2.9 → -2.8 | 0.0% → 0.0% | 110 → 115 |
| walk3_subject2 | 0 → 0 | -3.6 → -2.2 | 0.0% → 0.0% | 106 → 112 |
| walk3_subject5 | 1 → 0 | -8.2 → -5.2 | 4.0% → 1.3% | 148 → 151 |
| walk4_subject1 | 0 → 0 | -2.3 → -2.4 | 0.0% → 0.0% | 164 → 163 |

Flips in total 52 → 4, hand error +4 mm on average. Results trained on v1 (aiming1, run2, obstacles3,
dance2) stay in the table below marked v1, and all 14 clips are retrained on v3, so that the
comparison is made under one condition.

### Retargeting v3 — grounding the feet after the IK instead of weighting them

dance2 retrained on v2 got stuck from about 8,000 iterations at 61% of the clip (136 s), with the
adaptive sampler drawing that spot 70% of the time. It is a tuck jump that lifts the feet to 72 cm.
The human pelvis rises 0.47 m; the v2 reference asked for 0.59 m. With a knee that folds only to 131°,
the foot weight of 200 pulled the pelvis up to reach the foot targets. The setting that helped in
crouches inflated the jump.

v3 puts the foot weight back to the G1's 50 and fixes penetration after the IK
(`src/retarget_all.py --ground_lift`): if a foot collision box is below the floor, the whole body is
lifted by that much, with the lift spread over a 0.5 s window so it does not jump. This is the same
idea as the grounding correction in OmniRetarget and KungfuBot. mink's collision-avoidance limit was
tried first; it only stops a foot from going deeper, it does not push out one that is already under.

| | v2 | v3 |
|---|---|---|
| dance2 jump at 136 s (human 0.47 m, G1 0.42 m) | 0.59 m | 0.49 m |
| Lowest sole, 14 clips | -2.2 to -5.2 cm | 0.0 cm |
| Max root vertical speed (run2) | 2.07 m/s | 1.27 m/s |

The dance2 training on v2 (8,600 iterations) was stopped and restarted on v3.

dance2 trained to 30,000 iterations on v3 still completes 0%. The mean fall is at 137 s (61% of the clip),
the same place as on v1 (136 s), and the adaptive sampler sat near that tuck jump (129–137 s) throughout
training. It falls there even after the kinematic defects (flips, penetration, inflated jump) are gone.

So I measured whether the reference is **physically achievable**, with inverse dynamics
(`src/reference_qa.py`): compute the generalized forces without contact, explain the root wrench with
the forces the grounded sole points can push (upward only, friction pyramid), and compare the remaining
joint torques with the training torque limits. Over 128–142 s IGRIS-C asks the waist for up to 4.7x its
limit (at 129.2 s, a fast arm swing) and exceeds it in 24% of the frames; the G1 needs at most 1.3x, in
3%. The knee peaks at 1.1x and is not the bottleneck, so the earlier knee guess is withdrawn.

The likely reason is upper-body mass. Above the IGRIS-C waist sit 27.5 kg; just holding the torso fully
tilted takes 55 N·m, 92% of the waist limit (60 N·m in the vendor MJCF). The G1's is 15.9 kg and 29 N·m,
58% of its 50 N·m. IGRIS-C has almost no margin left for motion.

**Measured on the trained policy, though, this is not what ends dance2.** With `eval.py --torque_log`
recording joint torque saturation over 100 rollouts, the share at the limit just before the falls
(125–137 s) is 3.5% at the waist, 2.7% at the ankles and 0% at the knees. All 100 rollouts end at
137.9–138.0 s, the top of the tuck jump, when the whole reference body is in the air (lowest body 0.76 m
above the floor). The policy does not jump and tuck that far, and the 25 cm height-error termination
fires in every rollout at once. It barely uses its torque, so it is not running out of strength; it does
not attempt the jump. The waist and knee guesses are withdrawn for dance2. dance2 is not tuned further;
the walking clips train next.

Where the waist calculation matters is walking with a bent torso. In walk3_subject5 the torso leans 64°,
and IGRIS-C needs 106% of the waist limit just to **stand still** in that posture (G1: 52%). However the
knees or hips bend, the waist moment is the same while the torso is tilted, so retargeting cannot avoid
it. This is recorded as a prediction before training, to be checked against the result.

## All 14 clips from scratch (in progress)

IGRIS-C is being trained on the same 14 clips as the G1 teachers, the same way (BeyondMimic PPO, 30,000
iterations, from scratch). Evaluation: 100 rollouts from frame 0, domain randomization off. Rows are added
as clips finish (`scripts/igris/results_table.py`).

<!-- igris-table -->
| Clip | IGRIS reference | G1 success rate | IGRIS success rate | G1 mean survival | IGRIS mean survival | G1 E_mpbpe (mm) | IGRIS E_mpbpe (mm) | G1 E_mpjpe (rad) | IGRIS E_mpjpe (rad) |
|---|---|---|---|---|---|---|---|---|---|
| aiming1_subject1 | v1 | 100% | 100% | 100% | 100% | 35 | 36 | 0.080 | 0.091 |
| aiming1_subject1 | v3 | 100% | 100% | 100% | 100% | 35 | 37 | 0.080 | 0.088 |
| dance2_subject3 | v1 | 100% | 0% | 100% | 60% | 45 | — | 0.104 | — |
| dance2_subject3 | v3 | 100% | 0% | 100% | 61% | 45 | — | 0.104 | — |
| obstacles3_subject3 | v1 | 98% | 0% | 99% | 1% | 54 | — | 0.101 | — |
| run2_subject4 | v1 | 99% | 21% | 100% | 22% | 47 | 59 | 0.111 | 0.105 |
| walk1_subject1 | v3 | 99% | 100% | 99% | 100% | 34 | 35 | 0.066 | 0.073 |
| walk1_subject2 | v3 | 100% | 100% | 100% | 100% | 34 | 36 | 0.066 | 0.067 |
| walk1_subject5 | v3 | 100% | 99% | 100% | 100% | 34 | 36 | 0.069 | 0.073 |
| walk2_subject1 | v3 | 100% | 96% | 100% | 99% | 43 | 47 | 0.101 | 0.111 |
| walk2_subject3 | v3 | 96% | 96% | 98% | 98% | 50 | 49 | 0.104 | 0.116 |
| walk2_subject4 | v3 | 99% | 95% | 100% | 99% | 40 | 38 | 0.085 | 0.083 |
| walk3_subject2 | v3 | 100% | 100% | 100% | 100% | 34 | 37 | 0.071 | 0.078 |
| walk3_subject5 | v3 | 100% | 100% | 100% | 100% | 36 | 44 | 0.082 | 0.106 |
| walk4_subject1 | v3 | 100% | 100% | 100% | 100% | 37 | 27 | 0.062 | 0.067 |
<!-- /igris-table -->

walk1_subject2 was OOM-killed at iteration 27,441 and resumed from model_27000 to 30,000 (resuming resets only the adaptive sampler state). walk1_subject5 was resumed from 15,000 after a reboot.

run2_subject4 falls in the first 20 s, going from standing into a run, so it completes 21% from frame 0
(section above). The rollouts that fall do so about 4 s in, while accelerating from 0 to 2.9 m/s.

dance2_subject3 falls in all 100 rollouts, on average 136 s in (60% of the clip). The cause is the
retargeting, not the training. In the IGRIS-C reference the shoulder pitch and yaw jump about 1.5 rad
within one frame (1/30 s) in 33 frames (13 for the G1 on the same clip). With the arm raised, the IK seems
to switch to another solution that gives the same hand position, and the jump at 129 s comes right before
the mean fall time. jumps1_subject1 has 8 such frames, so it was moved to the end of the training order.
Both are retrained once the retargeting is fixed.

Checkpoints and evaluations: [Hugging Face `igris_c/`](https://huggingface.co/hooneyskywalker/humanoid-motion-tracking-policies/tree/main/igris_c) ·
training curves: [W&B `igris_c_transfer`, `igris_c_scratch`](https://wandb.ai/hooneyskywalker-humanoid/humanoid-motion-tracking)
