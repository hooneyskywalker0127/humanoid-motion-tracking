# 새 휴머노이드 추가 절차

[← README](../README.md) · [English](new-robot.en.md)

GMR 에 등록되지 않은 휴머노이드를 이 파이프라인에 올리는 순서입니다. IGRIS-C(ROBROS)와 ROBOTIS AI Sapiens K1 을
손으로 올리며 겪은 것을 스크립트 두 개와 점검표로 묶었습니다. 핵심은 **학습 전에 레퍼런스를 G1 과 같은 잣대로 채점**해서,
dance2 0%(10/17, 두 로봇 모두)처럼 18시간 학습 뒤에야 리타게팅 잘못을 알게 되는 일을 막는 것입니다.

```
A. 자동 설정   make_ik_config.py  MJCF → GMR IK json (매핑·스케일·오프셋·가중치)
B. 사람 점검   매핑 출력과 영자세 그림을 눈으로 확인, 벤더 값(무릎 토크) 입력
C. 채점        retarget_all.py 로 4개 클립 리타게팅 → scorecard.py 가 G1·사람과 비교해 통과/탈락
D. 학습        통과한 레퍼런스만 Isaac Lab 으로 (K1 파일을 본보기로 로봇 등록)
```

## A. GMR IK 설정 자동 생성

```bash
# GMR 환경, 저장소 루트
python scripts/newrobot/make_ik_config.py ~/Projects/GMR/assets/<robot>/<robot>.xml \
    --out configs/gmr/bvh_lafan1_to_<robot>_candidate.json --knee_torque <Nm> [--trunk_rot 100] [--map l_foot=<body> ...]
```

G1 설정(`bvh_lafan1_to_g1.json`)을 기준으로 네 가지를 채웁니다.

| 항목 | 방법 | 근거 |
|---|---|---|
| 링크 매핑 | 이름 규칙으로 pelvis·torso·thigh·shank·foot·shoulder·upper_arm·forearm·hand 를 찾음. 못 찾으면 오류, `--map 역할=몸체` 로 지정 | GMR 의 새 로봇 PR 들이 G1 json 을 복사해 이름만 바꾸는 방식 |
| 스케일 | 영자세 FK 로 다리(골반→발바닥)·팔(어깨→팔꿈치→손)·몸통(골반→어깨) 길이를 재고, G1 길이/G1 스케일(0.9/0.75/0.9)로 역산한 사람 길이로 나눔 | GMR `human_scale_table` 의 뜻 그대로. G1 에 돌리면 0.90/0.75/0.90 이 나옴 |
| 다리 스케일 상한 | 점프 능력 지수 τ_knee/(m·g·L_leg) 가 G1(0.52) 보다 작으면 다리·골반 스케일을 0.9 로 묶음 | IGRIS-C(0.29): 1.01 → dance2 0%, 0.90 → 100%. 무거운 로봇에 사람 크기 점프를 요구하면 안 됨 |
| 회전 오프셋 | G1 오프셋을 복사하되, 몸체 좌표계에서 본 다음 관절 방향이 G1 과 45° 넘게 다르면 그 차이(90° 단위로 반올림)만큼 돌림 | IGRIS 팔뚝은 −z, G1 은 +x → Ry(−90°). 손으로 찾던 값이 자동으로 나옴 |
| 가중치 | G1 표 + pelvis 위치 100, pelvis·torso 회전 `--trunk_rot`(기본 100), `ground_height` 0.03 | 몸통 가중치가 낮으면 팔다리 목표가 안 닿을 때 IK 가 몸통을 돌림(허우적) |

검증(10/19): IGRIS-C 자동 설정은 손으로 만든 retargeting v4 와 IK 표(가중치·오프셋)가 같고, dance2 를 다시 리타게팅한
채점 수치도 같습니다(체공 4 vs 5, 골반 +0.21 vs +0.22 m, 팔 방향 오차 3.5°/11.6° vs 3.5°/11.5°). K1 도 표 차이 0.
달라지는 것은 팔·몸통 스케일만으로, 손 설정은 G1 값(0.75/0.9)을 그대로 뒀고 자동은 실측(IGRIS 0.97/1.10, K1 0.53/0.95)입니다.

## B. 사람이 볼 것

1. 스크립트가 찍는 매핑 줄에서 발·손 몸체가 맞는지(발은 충돌 상자가 있는 몸체, 손은 GMR 가 손목으로 쓰는 몸체).
2. `--knee_torque` 는 MJCF 의 `ctrlrange` 가 아니라 벤더 스펙에서. MJCF 값이 비어 있거나 다른 로봇도 있습니다.
3. 오프셋 보정 줄이 찍힌 몸체는 영자세를 MuJoCo 뷰어로 한 번 봅니다. 팔꿈치가 쭉 펴진 채 굳으면 오프셋이 틀린 것입니다(IGRIS 10/2 사례).
4. 몸통 가중치: IGRIS 는 200, K1 은 100 으로 학습에 성공했습니다. 채점에서 flail 이 나오면 올립니다.

## C. 리타게팅과 채점

```bash
python src/retarget_all.py --src <bvh 폴더> --dst outputs/retarget_<robot> --robot <robot> \
    --ik_config configs/gmr/bvh_lafan1_to_<robot>_candidate.json \
    --posture 1.0 --contact_weight 100 20 --contact_threshold 0.08 --ground_lift --lift_window 1 --lift_sigma 0 \
    --vel_limit shoulder 8 --vel_limit elbow 8 --vel_limit wrist 8          # 생성기가 이 줄을 그대로 찍어 줍니다
python scripts/newrobot/scorecard.py <robot>:outputs/retarget_<robot>
```

채점은 클립마다 한 줄이고, 모든 항목이 문턱 안이면 "통과"입니다. 문헌의 절대 기준(PHUMA 관절 98%, NMR 관절 점프 0.5 rad,
비부유 1 cm)은 완주 100% 인 G1 레퍼런스도 넘기므로, **G1 과의 차이**(관절 포화·관절 점프·관통·부유·몸통 허우적·팔 방향·
루트 pitch)와 **사람과의 차이**(두 발 체공 비율·골반 최고 높이)로 잽니다. 문턱은 dance2 에서 0% 였던 레퍼런스(IGRIS E31,
K1 P10)와 100% 인 레퍼런스(IGRIS v4, K1 v1)가 갈리도록 잡았습니다. 자세한 정의와 문턱은 `scripts/newrobot/scorecard.py`
첫머리에 있습니다.

| 클립 | IGRIS E31 (0%) | IGRIS v4 (100%) | K1 P10 (0%) | K1 v1 (100%) |
|---|---|---|---|---|
| dance2 | 탈락 air +0.09, apex +0.07 | 통과 | 탈락 air +0.10, apex +0.08 | 통과 |
| obstacles3 | 탈락 air | 통과 + 무릎 경고 (81%) | 통과 + 무릎 경고 | 통과 + 무릎 경고 (94%) |
| jumps1 | 탈락 air, apex | 통과 | 통과 | 통과 |
| run2 | 탈락 air, pitch | **탈락 air +0.13, pitch 5%** | 탈락 air | 통과 |

무릎 경고(rom)는 사람 무릎 굽힘이 로봇 무릎 가동 범위를 넘는 프레임 비율이 2% 를 넘을 때 붙습니다. 운동학적으로 불가능한
구간이라 95% 미만을 예상하고, 리타게팅으로 고치는 대신 각주로 남깁니다(IGRIS 131° 4.1% → 81%, K1 135° 3.9% → 94%).
run2 의 IGRIS v4 탈락은 아직 학습 결과가 없는 **예측**입니다(10/19 저녁 결과로 문턱을 다시 봅니다).

탈락 항목별 손볼 곳: arm·pitch → 매핑/오프셋(A), air·apex·float → 스케일(A, `--no_cap` 끄지 말 것), pen·float →
`--contact_weight`/`--ground_lift`, jump·flail → `--vel_limit`/`--posture`/`--trunk_rot`.

## D. Isaac Lab 등록

K1 을 올린 파일을 그대로 본보기로 씁니다(fork `whole_body_tracking`).

- `robots/k1.py`: 로봇 cfg, 액추에이터 PD(G1 비율: kp = G1 kp/토크 비 × 새 로봇 토크, kd = 0.0637·kp), `*_ACTION_SCALE`,
  리타게팅 관절 순서. URDF 의 `package://` 경로는 `scripts/k1/prepare_urdf.py` 처럼 절대 경로로 바꿔 둡니다.
- `tasks/tracking/config/k1/`: `__init__.py`(태스크 등록), `flat_env_cfg.py`(몸체 이름·end-effector·원치 않는 접촉 정규식),
  `agents/rsl_rl_ppo_cfg.py`.
- `scripts/csv_to_npz.py --robot k1`: 레퍼런스 pkl → csv → npz.
- 학습·평가·영상은 `scripts/rsl_rl/server_loop.sh <robot> <seq> <ref> <csv 폴더>` 한 줄(30,000 iter, frame 0, 100 rollout, DR 끔).

## 아직 못 하는 것

- 영어 문서(`new-robot.en.md`)는 아직 없습니다.
- 채점 문턱은 두 로봇·4클립으로만 잡았습니다. run2·jumps1 결과가 나오면 다시 봅니다.
- 발·손 몸체 규칙은 이름 기반이라 이름이 특이한 로봇은 `--map` 이 필요합니다.
