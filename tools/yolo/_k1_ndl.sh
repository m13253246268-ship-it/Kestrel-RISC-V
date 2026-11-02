#!/bin/bash
# 复核 needle：重建引擎（main.c needle_hit 不区分大小写）后后台跑 Part F。
# 用法：先投送 src/main.c 与 /tmp/partf.sh；再次运行本脚本即轮询。
L=/tmp/ndl.log
if pgrep -f 'ndl_job' >/dev/null; then echo RUNNING; tail -15 $L; exit 0; fi
if grep -q NDL_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/ndl_job.sh <<'EOF'
cd /root/kv/kestrel_vllm_riscv || exit 1
cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_ndl
grep -c needle_hit src/main.c
sh tools/build/build_riscv.sh > /tmp/ndl_build.log 2>&1 || { tail -30 /tmp/ndl_build.log; echo BUILD_FAIL; echo NDL_END; exit 1; }
sha256sum src/main.c build-riscv/vllm_kestrel
timeout 7200 bash /tmp/partf.sh
echo NDL_END
EOF
nohup bash /tmp/ndl_job.sh > $L 2>&1 &
echo STARTED
