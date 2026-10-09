#!/usr/bin/env bash
# G1 | IGRIS-C 나란히(둘 다 프레임 0부터 끝까지) + 2배속 + 유튜브 메모를 그날 영상보관 G1vsIGRIS_<seq>/ 에 만든다.
#   bash scripts/igris/g1_vs_igris_video.sh walk1_subject2 [출력 폴더]
# 배치는 261006 G1vsIGRIS_walk1_subject1/compose_g1_igris_walk1.sh 와 같다. IGRIS 쪽은 마커 없는 체인 렌더.
set -e
SEQ=${1:?seq}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WBT=/home/sehoon/Projects/whole_body_tracking
G1=$(ls /home/sehoon/Desktop/참고/영상보관/g1-motion-tracking/09/*/$SEQ/compare/${SEQ}_policy.mp4 | head -1)
IG=$WBT/logs/igris_all/video/$SEQ.mp4
D=${2:-"/home/sehoon/Desktop/참고/영상보관/g1-motion-tracking/$(date +%m)/$(date +%y%m%d)/G1vsIGRIS_$SEQ"}  # 두 번째 인자로 출력 폴더를 바꿀 수 있다
mkdir -p "$D"
read G1C IGC G1E IGE DUR < <(python3 -c "
import json,math
g=json.load(open('$REPO/outputs/eval/$SEQ.json')); i=json.load(open('$WBT/logs/igris_all/eval_$SEQ.json'))
e=lambda d: '-' if d['e_mpbpe_mm'] is None or math.isnan(d['e_mpbpe_mm']) else str(round(d['e_mpbpe_mm']))
s=i['motion_frames']/50
print(round(g['success_rate']*100), round(i['success_rate']*100), e(g), e(i), f'{int(s//60)} min {round(s%60)} s')")
B=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; R=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
TITLE="Same Motion, Two Humanoids: Unitree G1 vs. IGRIS-C on LAFAN1 $SEQ (Isaac Lab)"
# 파일 이름 = 유튜브 제목(사용자 261006). 파일 이름에 못 쓰는 ':' 만 ' -' 로
OUT="$D/${TITLE//:/ -}.mp4"
ffmpeg -v error -y -i "$G1" -i "$IG" -filter_complex "
color=c=0x0e0f13:s=1920x1080:r=50[bg];
[0:v]scale=952:536[a];[1:v]scale=952:536[b];
[bg][a]overlay=4:300:shortest=1[t];[t][b]overlay=964:300:shortest=1,
drawtext=fontfile=$B:text='Same motion, two humanoids - LAFAN1 $SEQ':fontsize=44:fontcolor=white:x=(w-tw)/2:y=60,
drawtext=fontfile=$R:text='Same pipeline - GMR retargeting, BeyondMimic PPO from scratch, 30,000 iterations. Both from frame 0, full clip ($DUR).':fontsize=24:fontcolor=0xc8c8c8:x=(w-tw)/2:y=128,
drawtext=fontfile=$B:text='Unitree G1  (1.3 m, 35 kg)':fontsize=32:fontcolor=0x7fb0f0:x=24:y=214,
drawtext=fontfile=$R:text='29 joints  ·  completion $G1C% from frame 0':expansion=none:fontsize=22:fontcolor=0xb0b0b0:x=24:y=258,
drawtext=fontfile=$B:text='IGRIS-C  (1.5 m, 58 kg)':fontsize=32:fontcolor=0xf0a060:x=984:y=214,
drawtext=fontfile=$R:text='31 joints  ·  completion $IGC% from frame 0':expansion=none:fontsize=22:fontcolor=0xb0b0b0:x=984:y=258,
drawtext=fontfile=$R:text='Completion = share of 100 rollouts that reach the end without falling. Simulation only (Isaac Lab). Domain randomization off.':fontsize=22:fontcolor=0xa8a8a8:x=(w-tw)/2:y=900,
drawtext=fontfile=$B:text='%{eif\:t\:d}.%{eif\:mod(t*10\,10)\:d} s':fontsize=30:fontcolor=white:x=w-tw-28:y=896
[out]" -map "[out]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "$OUT"
# 2배속판은 화면에 2배속이라고 적는다(사용자 261006)
ffmpeg -v error -y -i "$OUT" -filter:v "setpts=0.5*PTS,drawtext=fontfile=$B:text='2x speed':fontsize=34:fontcolor=0xffd060:x=28:y=896" -r 50 -an -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "${OUT%.mp4} (2x speed).mp4"
cat > "$D/${TITLE//:/ -}_유튜브.md" <<EOF
# 유튜브 메모 — $(basename "$OUT") ($DUR, 2배속판 별도)

왼쪽 Unitree G1 | 오른쪽 IGRIS-C. 같은 LAFAN1 클립($SEQ), 둘 다 프레임 0(정지 상태)부터 끝까지.
G1 렌더: $G1
IGRIS-C 렌더: $IG (v3 레퍼런스, 마커 없음)
완주율: G1 $G1C/100, IGRIS-C $IGC/100 (100 롤아웃, 프레임 0 시작, 도메인 랜덤화 끔). 추적 오차(mpbpe): G1 $G1E mm, IGRIS-C $IGE mm.

제목

\`\`\`
$TITLE
\`\`\`

설명

\`\`\`
One motion-tracking pipeline, two humanoids, the same LAFAN1 clip ($SEQ, $DUR),
side by side from the same standing start.

Left: Unitree G1 (1.3 m, 35 kg, 29 joints). Right: IGRIS-C (1.5 m, 58 kg, 31 joints).

Each robot has its own policy, trained from scratch the same way: retarget the human motion
with GMR, then train PPO (BeyondMimic) for 30,000 iterations.

Completion over 100 rollouts from frame 0, domain randomization off:
  G1        $G1C of 100
  IGRIS-C   $IGC of 100

Mean per-body position error against the retargeted reference: G1 $G1E mm, IGRIS-C $IGE mm.
Simulation only (Isaac Lab).
\`\`\`
EOF
ls -la "$D"
