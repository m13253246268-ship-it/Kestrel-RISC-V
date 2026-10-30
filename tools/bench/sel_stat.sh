#!/bin/bash
# 复用率统计：过滤 decode（total=384）的 SELPROBE 行
cd ~/kv/spec || exit 1
grep -a 'SA-SELPROBE' sel.err | grep 'total=384' > d384.txt
echo "decode 样本数: $(wc -l < d384.txt)"
grep -oE 'reuse=[0-9.]+' d384.txt | sed 's/reuse=//' | awk '{s+=$1;n++;if(n==1||$1<mn)mn=$1;if($1>mx)mx=$1}END{printf "mean=%.4f min=%.4f max=%.4f\n",s/n,mn,mx}'
echo "--- 分布(前20):"
grep -oE 'reuse=[0-9.]+' d384.txt | sed 's/reuse=//' | head -20 | tr '\n' ' '
echo
echo "--- total=8 样本数(疑似prefill伪影):"
grep -a 'SA-SELPROBE' sel.err | grep 'total=8' | wc -l
echo "--- total 分布:"
grep -a 'SA-SELPROBE' sel.err | grep -oE 'total=[0-9]+' | sort | uniq -c
