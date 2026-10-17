# GMR IK 설정 (이 저장소가 추가한 로봇)

GMR 의 `general_motion_retargeting/ik_configs/` 에 넣고 `params.py` 의 `ROBOT_XML_DICT`, `IK_CONFIG_DICT["bvh_lafan1"]`,
`ROBOT_BASE_DICT`, `VIEWER_CAM_DISTANCE_DICT` 에 로봇 이름을 한 줄씩 더하면 된다.

| 파일 | 로봇 | 모델 | 만든 방법 |
|---|---|---|---|
| `bvh_lafan1_to_igris.json` | IGRIS-C (ROBROS) | `igris_c_description_public` MJCF | `scripts/igris/make_ik_config.py` (G1 표에서 링크 이름·팔뚝 회전·스케일을 바꿈) |
| `bvh_lafan1_to_k1.json` | AI Sapiens K1 (ROBOTIS) | `ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml` (meshdir 를 `meshes` 로) | G1 표 그대로, 손목 링크 이름만 `*_wrist_roll_rubber_hand` 로 |

K1 은 G1 과 치수가 거의 같다(골반→발바닥 0.797 vs 0.792 m, 질량 35.7 vs 34.4 kg, 영자세 팔 방향 같음). 그래서 스케일(다리 0.9, 팔 0.75)과
회전 오프셋을 바꾸지 않았다. walk1_subject2 리타게팅: 발 최저 중앙값 +0.9 cm, 관통 p1 −0.9 cm, 관절 한계 붙음 최대 1.8 %, 발목 roll 튐 12 프레임.

## 후보 설정 (10/17, dance2 재학습 중)

E31/P10(10/16) 은 dance2 에서 둘 다 0% 였다(IGRIS 137.9 s 턱점프에서 99/100 종료, K1 66 s·137.9 s). 원인은 레퍼런스가 G1 것보다
점프를 크게 요구한 것: 골반 최고 IGRIS +0.38~0.40 m, K1 +0.41 m(G1 +0.30, 사람 +0.33), 점프 전후 0.3 s 이상 체공 7~8회(G1 1회).
GMR 논문·코드엔 점프·접촉 처리가 없고, 골반 가중치 20 과 평활 ground_lift(창 15) 는 GMR 레시피 밖이라 되돌렸다.

| 파일 | 설정 | `src/retarget_all.py` 옵션 |
|---|---|---|
| `bvh_lafan1_to_igris_candidate.json` | 다리·골반 스케일 1.01→**0.90**(G1 과 같은 값), pelvis 위치 100, pelvis·torso 회전 200, ground_height 0.03 | `--posture 1.0 --vel_limit shoulder 8 --vel_limit elbow 8 --vel_limit wrist 8 --contact_weight 100 20 --contact_threshold 0.08 --ground_lift --lift_window 1 --lift_sigma 0` |
| `bvh_lafan1_to_k1_candidate.json` | G1 표, pelvis·torso 회전 100, ground_height 0.03 | 같은 옵션(속도 한도 10) |

채점(`scripts/igris/jump_scorecard.py`, dance2 130-140 s): IGRIS 골반 +0.22 m·체공 5회·허우적 0·jumps1 체공 0.47 s(사람 0.47),
K1 골반 +0.31 m·체공 2회·허우적 1·jumps1 0.40 s(G1 0.40). 역동역학 잔차 p95 IGRIS 0.58, K1 1.13(G1 1.09).
