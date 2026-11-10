#!/bin/bash
# Amendment 206d (2026-11-10, user: "improve this as much as we can before submission. the focus should be the novel pattern"): two more seeds
# of the cross-against-self pairs on G4, so that 206c's pair count has ten pairs, not six. 8 Colab G4 sessions with the frozen job script of
# 206 (runs/xql/job206/xq_online.sh; code7; keep 1.0, replay every 4, 600 items, the 240-point gate), in this order as slots free, for seed 4,
# then 5:
#   xqgSCx<s>  S4  the 4x self-corrector   TEACH=control CORR=4x
#   xqgDx<s>   D4  the 4x cross corrector  TEACH=bridge  CORR=4x
#   xqgSC<s>   S   the 1x self-corrector   TEACH=control CORR=1x
#   xqgD<s>    D   the 1x cross corrector  TEACH=bridge  CORR=1x
# Starts nothing until 206's launcher has ended (XQ_ONL11_DONE in the log, or its process gone with none of its sessions starting or running):
# onl11 counts only its own sessions, so the two launchers must not overlap (at most CAP = 3 sessions in all). G4 only (6 tries, 10 minutes
# apart). Lock and balance guard (30 CU per session) as before. No start after 21:30 on Nov 10.
# Run from the Mac:  kaggle_g4agent/dreamer/detach.sh /tmp/xq_onl12.out bash kaggle_g4agent/dreamer/xq_onl12_launch.sh
# Log runs/xql/side.log; ends with XQ_ONL12_DONE. Never kill this launcher's process group while its sessions run (it holds their pollers).
K=~/systemone/kaggle_g4agent; R=$K/runs; CR=~/systemone/v2/infra/colab/colab_run.sh; L=$R/xql/side.log; LOCK=/tmp/xq_slot_lock; MIN_CU=${MIN_CU:-30}; CAP=${CAP:-3}
log() { echo "$(date +%H:%M) $*" >> $L; }
ARMS="SCx Dx SC D"; OLD=""; for s in 1 2 3; do for a in SC D SCx Dx IM C; do OLD="$OLD xqg$a$s"; done; done; NEW=""; for s in 4 5; do for a in $ARMS; do NEW="$NEW xqg$a$s"; done; done
SESS="xqoC1 xqoC2 xqoC3 xqoD1 xqoD3 xqoG1 xqoG2 xqoG3 xqoD4x xqoS1 xqoF1 xqo204 xqoSC1 xqoD2 xqoBL1$OLD$NEW"
running() { local n=0; for s in $SESS; do grep -qE "^(starting|running)" $R/$s/status 2>/dev/null && n=$((n + 1)); done; echo $n; }
balance() { local b k; for k in 1 2 3; do
    b=$( (cd ~ && /usr/bin/perl -e 'alarm 120; exec @ARGV' ~/.local/bin/colab --auth oauth2 usage 2>/dev/null) | sed -n 's/^Current balance: \([0-9.]*\).*/\1/p' | head -n 1)
    [ -n "$b" ] && { echo $b; return; }; sleep 30; done; }
start() { local g=$1 s=$2 d=$3; shift 3; mkdir -p $R/$s; echo starting > $R/$s/status
  ( JOB_DESC="$d" $CR -s $s -g $g -o /content/out -d $R/$s "$@" > $R/$s/colab_run.out 2>&1; rc=$?
    grep -qE "^exit" $R/$s/status 2>/dev/null || echo "exit $rc launcher (colab_run ended before writing a status)" > $R/$s/status
    log "$s ended: $(cat $R/$s/status)" ) &
  log "started $s on $g ($d)"; }
settled() {   # review 204: started a job, gave up, or ended for good (onl8 retries after "no runtime" and exit 4 ends, and counts none of these four)
  grep -q "started xq_online.sh" $R/$1/run.log 2>/dev/null && return 0; grep -q "$1: gave up" $L && return 0
  grep -qE "^exit" $R/$1/status 2>/dev/null && ! grep -qE "^exit (3 no runtime|4 )" $R/$1/status; }
LATE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 21:30:00" +%s)
cap_now() { if settled xqoF1 || grep -q XQ_ONL8_DONE $L || ! pgrep -f xq_onl8_launch.sh > /dev/null; then echo $CAP; else echo $((CAP - 1)); fi; }   # a slot stays free for xqoF1 until it has settled (or its launcher is gone)
start_when_free() { local told=""
  while true; do
    [ $(date +%s) -gt $LATE ] && { log "$2: not started by its cutoff ($(date -r $LATE +%H:%M))"; return 1; }
    if mkdir $LOCK 2>/dev/null; then
      local n=$(running) c=$(cap_now)
      if [ $n -lt $c ]; then
        local b=$(balance) need=$((MIN_CU * (n + 1)))
        if [ -n "$b" ] && awk "BEGIN{exit !($b >= $need)}"; then start "$@"; sleep 30; rmdir $LOCK; return 0; fi
        [ -z "$told" ] && log "balance '${b:-unknown}' below $need CU ($n running): $2 waits"; told=1; rmdir $LOCK; sleep 600; continue
      fi
      rmdir $LOCK
    fi
    sleep 60
  done
}
run_g4() {   # SESSION DESC CMD...: a G4 only; 6 tries, 10 minutes apart; none after $LATE
  local s=$1 r f
  for r in 1 2 3 4 5 6; do
    f=1; while [ -e $R/${s}_f$f ]; do f=$((f + 1)); done
    [ -d $R/$s ] && mv $R/$s $R/${s}_f$f
    start_when_free G4 "$@" || return 1
    until grep -qE "^exit" $R/$s/status 2>/dev/null; do sleep 60; done
    if grep -qE "^exit (3 no runtime|4 )" $R/$s/status; then log "$s: no G4 session ($(cat $R/$s/status)), try $r"; [ $r -lt 6 ] && sleep 600; continue; fi
    return 0
  done
  log "$s: gave up after 6 tries (G4)"; }
begun() { grep -qE "^(starting|running|exit)" $R/$1/status 2>/dev/null; }
after() {   # SESSION PID: wait until the session's job has started or its waiter has ended (review 205: "starting" alone let a failed start lose
  local seen=""   # its slot to the next session); gives up 45 min after the session's first start attempt, so a hung start cannot hold the chain
  until grep -q "started xq_online.sh" $R/$1/run.log 2>/dev/null || ! kill -0 $2 2>/dev/null; do
    if begun $1; then [ -z "$seen" ] && seen=$(date +%s)
      [ $(( $(date +%s) - seen )) -gt 2700 ] && { log "$1: no job 45 min after its first start attempt; the next session's waiter starts"; break; }; fi
    sleep 60; done; }
log "onl12 launcher: 206d, seeds 4-5 of the cross-against-self pairs on G4 (S4, D4, S, D), at most $CAP sessions; waits for 206's launcher to end"
J=$R/xql/job206/xq_online.sh; [ -f $J ] || { log "onl12: the frozen job script $J is missing: not started"; exit 1; }
old_busy() { local s; for s in $OLD; do grep -qE "^(starting|running)" $R/$s/status 2>/dev/null && return 0; done; return 1; }
until grep -qE "^[0-9]{2}:[0-9]{2} XQ_ONL11_DONE$" $L || { ! pgrep -f xq_onl11_launch.sh > /dev/null && ! old_busy; }; do   # onl11 counts only its own sessions: no overlap
  [ $(date +%s) -gt $LATE ] && { log "onl12: 206's launcher had not ended by its cutoff ($(date -r $LATE +%H:%M)); nothing started"; log "XQ_ONL12_DONE"; exit 0; }
  sleep 120; done
log "onl12: 206's launcher has ended; starting seeds 4-5"
for s in 4 5; do for a in $ARMS; do
  case $a in
    SC)  d="self-corrector";               args="TEACH=control CODE=code7 CORR=1x";;
    D)   d="setup D, the cross corrector"; args="TEACH=bridge CODE=code7 CORR=1x";;
    SCx) d="4x self-corrector";            args="TEACH=control CODE=code7 CORR=4x";;
    Dx)  d="setup D, the 4x corrector";    args="TEACH=bridge CODE=code7 CORR=4x";;
  esac
  n=xqg$a$s
  run_g4 $n "206d $d + keep + replay, seed $s, 600 items + gate (G4)" $J ON_SEED=$s $args & p=$!
  after $n $p
done; done
wait; log "XQ_ONL12_DONE"
