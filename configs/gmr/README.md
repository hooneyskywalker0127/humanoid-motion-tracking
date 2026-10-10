# GMR IK 설정 (이 저장소가 추가한 로봇)

GMR 의 `general_motion_retargeting/ik_configs/` 에 넣고 `params.py` 의 `ROBOT_XML_DICT`, `IK_CONFIG_DICT["bvh_lafan1"]`,
`ROBOT_BASE_DICT`, `VIEWER_CAM_DISTANCE_DICT` 에 로봇 이름을 한 줄씩 더하면 된다.

| 파일 | 로봇 | 모델 | 만든 방법 |
|---|---|---|---|
| `bvh_lafan1_to_igris.json` | IGRIS-C (ROBROS) | `igris_c_description_public` MJCF | `scripts/igris/make_ik_config.py` (G1 표에서 링크 이름·팔뚝 회전·스케일을 바꿈) |
| `bvh_lafan1_to_k1.json` | AI Sapiens K1 (ROBOTIS) | `ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml` (meshdir 를 `meshes` 로) | G1 표 그대로, 손목 링크 이름만 `*_wrist_roll_rubber_hand` 로 |

K1 은 G1 과 치수가 거의 같다(골반→발바닥 0.797 vs 0.792 m, 질량 35.7 vs 34.4 kg, 영자세 팔 방향 같음). 그래서 스케일(다리 0.9, 팔 0.75)과
회전 오프셋을 바꾸지 않았다. walk1_subject2 리타게팅: 발 최저 중앙값 +0.9 cm, 관통 p1 −0.9 cm, 관절 한계 붙음 최대 1.8 %, 발목 roll 튐 12 프레임.
