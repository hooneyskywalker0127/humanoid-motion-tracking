#!/usr/bin/env bash
# 14클립 G1 | IGRIS-C 나란히 영상을 하나로 잇는다(+ 2배속, 유튜브 메모). 클립별 영상은 g1_vs_igris_video.sh 로 만든 것.
#   bash scripts/igris/g1_vs_igris_14clips.sh
# 쉬운 것에서 어려운 것 순서. 클립별 영상이 없거나 마커 있는 옛 판이면 작업 폴더에 새로 만든다.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WBT=/home/sehoon/Projects/whole_body_tracking
ARCH=/home/sehoon/Desktop/참고/영상보관/g1-motion-tracking
ORDER="walk1_subject1 walk1_subject2 walk1_subject5 walk2_subject1 walk2_subject3 walk2_subject4 walk3_subject2 walk3_subject5 walk4_subject1 aiming1_subject1 run2_subject4 dance2_subject3 obstacles3_subject3 jumps1_subject1"
TITLE="Same Motion, Two Humanoids: Unitree G1 vs. IGRIS-C on 14 LAFAN1 Clips (Isaac Lab)"
D="$ARCH/$(date +%m)/$(date +%y%m%d)/G1vsIGRIS_LAFAN1_14clips"; W="$D/clips"; mkdir -p "$W"
LIST="$W/list.txt"; : > "$LIST"
for s in $ORDER; do
  f=$(ls "$ARCH"/10/*/G1vsIGRIS_$s/"Same Motion, Two Humanoids - Unitree G1 vs. IGRIS-C on LAFAN1 $s (Isaac Lab).mp4" 2>/dev/null | tail -1)
  # walk1_subject1 의 기존 세트(261006)는 마커 있는 IGRIS 렌더라 새로 만든다
  if [[ -z $f || $s == walk1_subject1 ]]; then bash "$REPO/scripts/igris/g1_vs_igris_video.sh" $s "$W/$s" > /dev/null
    f="$W/$s/Same Motion, Two Humanoids - Unitree G1 vs. IGRIS-C on LAFAN1 $s (Isaac Lab).mp4"; fi
  echo "file '$f'" >> "$LIST"
done
OUT="$D/${TITLE//:/ -}.mp4"
ffmpeg -v error -y -f concat -safe 0 -i "$LIST" -c copy "$OUT"
B=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ffmpeg -v error -y -i "$OUT" -filter:v "setpts=0.5*PTS,drawtext=fontfile=$B:text='2x speed':fontsize=34:fontcolor=0xffd060:x=28:y=896" -r 50 -an -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p "${OUT%.mp4} (2x speed).mp4"
ROWS=$(python3 - <<PY
import json,math
for s in "$ORDER".split():
    g=json.load(open(f"$REPO/outputs/eval/{s}.json")); i=json.load(open(f"$WBT/logs/igris_all/eval_{s}.json"))
    print(f"  {s:<20} G1 {round(g['success_rate']*100):>3}%   IGRIS-C {round(i['success_rate']*100):>3}%")
PY
)
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | python3 -c "import sys;s=float(sys.stdin.read());print(f'{int(s//60)} min {round(s%60)} s')")
cat > "$D/${TITLE//:/ -}_유튜브.md" <<MD
# 유튜브 메모 — $(basename "$OUT") ($DUR, 2배속판 별도)

14클립을 쉬운 것에서 어려운 것 순으로 이었다. 각 클립: 왼쪽 G1 | 오른쪽 IGRIS-C, 둘 다 frame 0부터 끝까지, IGRIS-C 는 v3 Reference·마커 없음.
Success Rate: 100 rollout, frame 0 시작, Domain Randomization 끔.

제목

\`\`\`
$TITLE
\`\`\`

설명

\`\`\`
One motion-tracking pipeline, two humanoids, the same 14 LAFAN1 clips, each side by side from the
same standing start. Left: Unitree G1 (1.3 m, 35 kg, 29 joints). Right: IGRIS-C (1.5 m, 58 kg, 31 joints).

Each robot has its own policy per clip, trained from scratch the same way: retarget the human
motion with GMR, then train PPO (BeyondMimic) for 30,000 iterations.

Success rate over 100 rollouts from frame 0, domain randomization off:
$ROWS

All nine walking clips transfer to IGRIS-C at the G1's level. The jumping, running, dancing and
crawling clips do not yet; comparing the references, the IGRIS-C retargeting lifts the feet more
often and for longer than the human, and does not get down to the floor in the crawl.
Simulation only (Isaac Lab).
\`\`\`
MD
ls -la "$D"
