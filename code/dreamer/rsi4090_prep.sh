#!/bin/bash
# 177 on the 4090 (user 11-07: "dont run the rsi on the colab, use the 4090 with the smaller models that fit there"): inputs for the Qwen3-1.7B
# RSI pilot in ~/rsi4090. LiveCodeBench (granted) file by file, kept to easy and medium (the only rows rsi.py reads), the original deleted at
# once (the disk is 99% full); Qwen3-1.7B (granted); the s3 / question rows and E4B's verified rows from the private HF repo; the merged off-task
# rows. Stops if fewer than 4 GB stay free. Log ~/rsi4090/prep.log; PREP_DONE.
D=~/rsi4090; HF=~/toyq/.pt/bin/hf; P=~/toyq/.pt/bin/python; L=$D/prep.log; mkdir -p $D/lcb $D/raw $D/inx $D/xb
log() { echo "$(date +%H:%M) $*" >> $L; }
room() { local f=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$f" -ge 4 ] || { log "only ${f} GB free: stopping"; exit 1; }; log "free ${f} GB"; }
room
for f in test test2 test3 test4 test5 test6; do
  [ -f $D/lcb/$f.jsonl ] && continue
  $HF download livecodebench/code_generation_lite --repo-type dataset --revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505 --include "$f.jsonl" --local-dir $D/raw > /dev/null 2>&1 || { log "download $f FAILED"; exit 1; }
  $P - $D/raw/$f.jsonl $D/lcb/$f.jsonl.part <<'PY' >> $L 2>&1 || { log "filter $f FAILED"; exit 1; }
import json, sys
n = k = 0
with open(sys.argv[1]) as fi, open(sys.argv[2], "w") as fo:
    for line in fi:
        n += 1
        if json.loads(line)["difficulty"] in ("easy", "medium"):
            fo.write(line); k += 1
print(sys.argv[1].rsplit("/", 1)[1], "rows", n, "kept (easy, medium)", k)
PY
  mv $D/lcb/$f.jsonl.part $D/lcb/$f.jsonl; rm -f $D/raw/$f.jsonl; room
done
rm -rf $D/raw
[ -f $D/mqwen/config.json ] || $HF download Qwen/Qwen3-1.7B --local-dir $D/mqwen > /dev/null 2>&1 || { log "Qwen download FAILED"; exit 1; }; room
$HF download SkyPanther/g4agent-job --repo-type dataset --include "s3/*" --include "xq/*.npy" --include "xq/*.jsonl" --include "xq/learn/xl_rows.jsonl" --local-dir $D/inx > /dev/null 2>&1 || { log "inputs download FAILED"; exit 1; }
cd $D/bench && $P mix_merge.py $D/inx/s3 $D/inx/xq $D/xb >> $L 2>&1 && cp $D/inx/xq/xq_questions.jsonl $D/xb/ || { log "merge FAILED"; exit 1; }
log "lcb: $(du -sh $D/lcb | cut -f1) | qwen: $(du -sh $D/mqwen | cut -f1) | E4B rows: $(wc -l < $D/inx/xq/learn/xl_rows.jsonl)"; room; log "PREP_DONE"
