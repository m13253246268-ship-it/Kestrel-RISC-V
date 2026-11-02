#!/bin/bash
# P3：QPAR（激活量化并行）+ SCV（scale vfwcvt）重建 + 位级/NLL/性能 A/B。
# 后台跑；再次运行即轮询。日志 /tmp/p3.log
L=/tmp/p3.log
if pgrep -f 'p3_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q P3_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/p3_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_p3 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_p3
sha256sum src/model/vllm_safetensors.c
sh tools/build/build_riscv.sh > /tmp/p3_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE 'error|vfwcvt|zvfh' /tmp/p3_build.log | head -5
tail -2 /tmp/p3_build.log
B=build-riscv/vllm_kestrel
sha256sum $B
echo "== colbt_worker disasm =="
objdump -d --no-show-raw-insn $B | awk '/<rv_q4_colbt_worker[^>]*>:$/{p=1;next} p&&/^$/{p=0} p' > /tmp/p3_w.s
echo "lines=$(wc -l < /tmp/p3_w.s) vfwcvt.f.f=$(grep -c 'vfwcvt.f.f.v' /tmp/p3_w.s) vlse16=$(grep -c vlse16 /tmp/p3_w.s)"
R=/root/kv/models
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
arm() { case $1 in base) echo "";; qpar) echo "VLLM_Q4_QPAR=1";; scv) echo "VLLM_Q4_SCV=1";; both) echo "VLLM_Q4_QPAR=1 VLLM_Q4_SCV=1";; esac; }
echo "== 位级 + NLL（Qwen3-0.6B ctx32 t8）=="
for A in base qpar scv both; do
  O=/tmp/p3_chk_$A.log
  env $COLB $(arm $A) VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/Qwen3-0.6B-col --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
  echo "-- $A rc=$? colbt_bad=$(grep -c 'Q4CHK-colbt' $O)/$(grep 'Q4CHK-colbt' $O | grep -vc 'bitmismatch=0') qpar_lines=$(grep -c 'Q4CHK-qpar' $O) qpar_bad=$(grep 'Q4CHK-qpar' $O | grep -vc 'q_mismatch=0 d_mismatch=0')"
  grep 'CORPUS_NLL_MEAN' $O
  grep -E '^\[STREAM\] (out|text)' $O | md5sum
done
echo "== 性能（t8，每臂 2 轮）=="
for M in Qwen3-0.6B Qwen3-4B; do for X in 32 128; do for r in 1 2; do for A in base qpar scv both; do
  O=/tmp/p3_perf.log
  env $COLB $(arm $A) VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx $X --stream-n 16 --threads 8 --warmup > $O 2>&1
  printf '%s ctx=%s r%s %-4s rc=%s load=%s | ' $M $X $r $A $? "$(cut -d' ' -f1 /proc/loadavg)"
  grep -E '^\[STREAM\] prefill' $O | tail -1 | tr '\n' ' '
  grep '\[PREFILL-TIMING\]' $O | tail -1 | cut -c1-200
  echo " txt=$(grep -E '^\[STREAM\] (out|text)' $O | md5sum | cut -c1-8)"
done; done; done; done
uptime
echo P3_END
EOF
nohup bash /tmp/p3_job.sh > $L 2>&1 &
echo STARTED
