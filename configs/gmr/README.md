# GMR IK 설정 (이 저장소가 추가한 로봇)

GMR 의 `general_motion_retargeting/ik_configs/` 에 넣고 `params.py` 의 `ROBOT_XML_DICT`, `IK_CONFIG_DICT["bvh_lafan1"]`,
`ROBOT_BASE_DICT`, `VIEWER_CAM_DISTANCE_DICT` 에 로봇 이름을 한 줄씩 더하면 된다.

| 파일 | 로봇 | 모델 | 만든 방법 |
|---|---|---|---|
| `bvh_lafan1_to_igris.json` | IGRIS-C (ROBROS) | `igris_c_description_public` MJCF | `scripts/igris/make_ik_config.py` (G1 표에서 링크 이름·팔뚝 회전·스케일을 바꿈) |
| `bvh_lafan1_to_k1.json` | AI Sapiens K1 (ROBOTIS) | `ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml` (meshdir 를 `meshes` 로) | G1 표 그대로, 손목 링크 이름만 `*_wrist_roll_rubber_hand` 로 |

K1 은 G1 과 치수가 거의 같다(골반→발바닥 0.797 vs 0.792 m, 질량 35.7 vs 34.4 kg, 영자세 팔 방향 같음). 그래서 스케일(다리 0.9, 팔 0.75)과
회전 오프셋을 바꾸지 않았다. walk1_subject2 리타게팅: 발 최저 중앙값 +0.9 cm, 관통 p1 −0.9 cm, 관절 한계 붙음 최대 1.8 %, 발목 roll 튐 12 프레임.

## 후보 설정 (10/10, 학습 전)

| 파일 | 바뀐 것 | 함께 쓰는 `src/retarget_all.py` 옵션 |
|---|---|---|
| `bvh_lafan1_to_igris_candidate.json` | 2단계 pelvis 위치 100→20, pelvis·torso 회전 5/10→50(두 표) | `--contact_weight 100 20 --contact_threshold 0.08 --ground_lift --posture 1.0 --posture_zero 20` |
| `bvh_lafan1_to_k1_candidate.json` | 같은 변경 | 같은 옵션 |

이유: 기본 가중치에서는 팔 목표가 안 닿을 때 IK 가 몸통을 돌리고, 어깨가 ±π 로 감긴 등가 해에 갇혔다가 한 프레임에 풀리면서
몸통이 1000°/s 로 튄다(dance2 129 s, run2 199 s). `scripts/igris/ref_smoothness.py` 로 잰다(사람 뼈 대비 링크 각속도).
대가: 점프 체공이 사람보다 길어진다(jumps1 0.93 s vs 사람 0.47 s). 최종 판정은 재학습으로만.
