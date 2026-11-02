#!/bin/bash
# P2：IME vmadot 接入（VLLM_Q4_IME=1 混合 / =2 仅 IME）+ P3 默认开启验证。
# 后台跑；再次运行即轮询。日志 /tmp/p2.log
L=/tmp/p2.log
if pgrep -f 'p2_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q P2_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/p2_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_p2 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_p2
sha256sum src/model/vllm_safetensors.c
sh tools/build/build_riscv.sh > /tmp/p2_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE 'error|warning: .*ime' /tmp/p2_build.log | head -8
tail -2 /tmp/p2_build.log
B=build-riscv/vllm_kestrel
sha256sum $B
echo "== rv_ime_q4_tile disasm =="
objdump -d $B | awk '/<rv_ime_q4_tile[^>]*>:$/{p=1;next} p&&/^$/{p=0} p' > /tmp/p2_t.s
echo "lines=$(wc -l < /tmp/p2_t.s) insn_2b=$(grep -cE '\s[0-9a-f]{6}2b\s' /tmp/p2_t.s) vfwcvt=$(grep -c vfwcvt /tmp/p2_t.s)"
R=/root/kv/models
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
arm() { case $1 in def) echo "";; off) echo "VLLM_Q4_QPAR=0 VLLM_Q4_SCV=0";; ime1) echo "VLLM_Q4_IME=1";; ime2) echo "VLLM_Q4_IME=2";; esac; }
echo "== 位级 + NLL（Qwen3-0.6B ctx32 t8）=="
for A in def off ime1 ime2; do
  O=/tmp/p2_chk_$A.log
  env $COLB $(arm $A) VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/Qwen3-0.6B-col --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
  echo "-- $A rc=$? colbt=$(grep -c 'Q4CHK-colbt' $O)/bad$(grep 'Q4CHK-colbt' $O | grep -vc 'bitmismatch=0') qpar=$(grep -c 'Q4CHK-qpar' $O)/bad$(grep 'Q4CHK-qpar' $O | grep -vc 'q_mismatch=0 d_mismatch=0') ime=$(grep -c 'Q4CHK-ime' $O)/bad$(grep 'Q4CHK-ime' $O | grep -vc 'bitmismatch=0')"
  grep 'Q4CHK-ime' $O | head -2
  grep -E 'SIGILL|Illegal|Segmentation' $O | head -2
  grep 'Q4IME' $O | tail -1
  grep 'CORPUS_NLL_MEAN' $O
  echo "txt=$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8) n=$(grep -ac '\[STREAM\] text:' $O)"
done
echo "== 性能（t8，每臂 2 轮）=="
for M in Qwen3-0.6B Qwen3-4B; do for X in 32 128; do for r in 1 2; do for A in def ime1 ime2; do
  O=/tmp/p2_perf.log
  env $COLB $(arm $A) VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx $X --stream-n 16 --threads 8 --warmup > $O 2>&1
  printf '%s ctx=%s r%s %-4s rc=%s load=%s | ' $M $X $r $A $? "$(cut -d' ' -f1 /proc/loadavg)"
  grep -a '\[STREAM\] prefill' $O | tail -1 | tr '\n' ' '
  grep -a '\[PREFILL-TIMING\]' $O | tail -1 | cut -c1-200
  echo " $(grep -a 'Q4IME' $O | tail -1) txt=$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
done; done; done; done
uptime
echo P2_END
EOF
nohup bash /tmp/p2_job.sh > $L 2>&1 &
echo STARTED
