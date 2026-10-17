#!/usr/bin/env bash
# G1 | <로봇> 나란히(둘 다 프레임 0부터 끝까지) + 2배속 + 유튜브 메모. g1_vs_igris_video.sh 를 로봇·레퍼런스 버전 인자로 일반화한 판.
#   bash scripts/igris/g1_vs_robot_video.sh <seq> <igris|k1> <ref 태그: e31|p10> [출력 폴더]
# 로봇 쪽 렌더·평가는 server_loop.sh 산출(logs/server_<robot>_<ref>/video/<seq>.mp4, eval_<seq>.json).
set -e
SEQ=${1:?seq}; ROBOT=${2:?igris|k1}; REF=${3:?ref}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WBT=/home/sehoon/Projects/whole_body_tracking
case $ROBOT in
  igris) NAME="IGRIS-C"; SPEC="1.5 m, 58 kg"; JOINTS=31; FOLDER=G1vsIGRIS; YT="IGRIS-C" ;;
  k1)    NAME="ROBOTIS AI Sapiens K1"; SPEC="1.2 m, 36 kg"; JOINTS=23; FOLDER=G1vsK1; YT="ROBOTIS AI Sapiens K1" ;;
esac
G1=$(ls /home/sehoon/Desktop/참고/영상보관/g1-motion-tracking/09/*/$SEQ/compare/${SEQ}_policy.mp4 | head -1)
RV=$WBT/logs/server_${ROBOT}_$REF/video/$SEQ.mp4
EV=$WBT/logs/server_${ROBOT}_$REF/eval_$SEQ.json
D=${4:-"/home/sehoon/Desktop/참고/영상보관/g1-motion-tracking/$(date +%m)/$(date +%y%m%d)/${FOLDER}_$SEQ"}
mkdir -p "$D"
read G1C RC G1E RE DUR ALIVE < <(python3 -c "
import json,math
g=json.load(open('$REPO/outputs/eval/$SEQ.json')); i=json.load(open('$EV'))
e=lambda d: '-' if d['e_mpbpe_mm'] is None or math.isnan(d['e_mpbpe_mm']) else str(round(d['e_mpbpe_mm']))
s=i['motion_frames']/50
print(round(g['success_rate']*100), round(i['success_rate']*100), e(g), e(i), f'{int(s//60)}min{round(s%60)}s', round(i['mean_alive_frames']/50))")
DUR=${DUR/min/ min }; DUR=${DUR/s/ s}
B=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; R=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
TITLE="Same Motion, Two Humanoids: Unitree G1 vs. $YT on LAFAN1 $SEQ (Isaac Lab)"
OUT="$D/${TITLE//:/ -}.mp4"
ffmpeg -v error -y -i "$G1" -i "$RV" -filter_complex "
color=c=0x0e0f13:s=1920x1080:r=50[bg];
[0:v]scale=952:536[a];[1:v]scale=952:536[b];
[bg][a]overlay=4:300:shortest=1[t];[t][b]overlay=964:300:shortest=1,
drawtext=fontfile=$B:text='Same motion, two humanoids - LAFAN1 $SEQ':fontsize=44:fontcolor=white:x=(w-tw)/2:y=60,
drawtext=fontfile=$R:text='Same pipeline - GMR retargeting, BeyondMimic PPO from scratch, 30,000 iterations. Both from frame 0, full clip ($DUR).':fontsize=24:fontcolor=0xc8c8c8:x=(w-tw)/2:y=128,
drawtext=fontfile=$B:text='Unitree G1  (1.3 m, 35 kg)':fontsize=32:fontcolor=0x7fb0f0:x=24:y=214,
drawtext=fontfile=$R:text='29 joints  ·  Success Rate $G1C% from frame 0':expansion=none:fontsize=22:fontcolor=0xb0b0b0:x=24:y=258,
drawtext=fontfile=$B:text='$NAME  ($SPEC)':fontsize=32:fontcolor=0xf0a060:x=984:y=214,
drawtext=fontfile=$R:text='$JOINTS joints  ·  Success Rate $RC% from frame 0  ·  reference $REF':expansion=none:fontsize=22:fontcolor=0xb0b0b0:x=984:y=258,
drawtext=fontfile=$R:text='Success Rate = share of 100 rollouts that reach the end without a termination. Simulation only (Isaac Lab). Domain randomization off.':fontsize=22:fontcolor=0xa8a8a8:x=(w-tw)/2:y=900,
drawtext=fontfile=$B:text='%{eif\:t\:d}.%{eif\:mod(t*10\,10)\:d} s':fontsize=30:fontcolor=white:x=w-tw-28:y=896
[out]" -map "[out]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "$OUT"
ffmpeg -v error -y -i "$OUT" -filter:v "setpts=0.5*PTS,drawtext=fontfile=$B:text='2x speed':fontsize=34:fontcolor=0xffd060:x=28:y=896" -r 50 -an -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "${OUT%.mp4} (2x speed).mp4"
cat > "$D/${TITLE//:/ -}_유튜브.md" <<MEMO
# 유튜브 메모 — $(basename "$OUT") ($DUR, 2배속판 별도)

왼쪽 Unitree G1 | 오른쪽 $NAME. 같은 LAFAN1 클립($SEQ), 둘 다 프레임 0(정지 상태)부터 끝까지.
G1 렌더: $G1
$NAME 렌더: $RV (레퍼런스 $REF, 마커 없음)
Success Rate: G1 $G1C/100, $NAME $RC/100 (100 롤아웃, 프레임 0 시작, 도메인 랜덤화 끔). 평균 생존 ${ALIVE} s / $DUR. 추적 오차(mpbpe): G1 $G1E mm, $NAME $RE mm.

제목

\`\`\`
$TITLE
\`\`\`

설명

\`\`\`
One motion-tracking pipeline, two humanoids, the same LAFAN1 clip ($SEQ, $DUR),
side by side from the same standing start.

Left: Unitree G1 (1.3 m, 35 kg, 29 joints). Right: $NAME ($SPEC, $JOINTS joints).

Each robot has its own policy, trained from scratch the same way: retarget the human motion
with GMR (reference $REF), then train PPO (BeyondMimic) for 30,000 iterations.

Success Rate over 100 rollouts from frame 0, domain randomization off:
  G1   $G1C of 100
  $NAME   $RC of 100   (mean survival ${ALIVE} s of $DUR)

Mean per-body position error against the retargeted reference: G1 $G1E mm, $NAME $RE mm.
Simulation only (Isaac Lab).
\`\`\`
MEMO
ls -la "$D"
