#!/bin/bash
pkill -f q3cmp_job.sh; pkill -f llama-bench; pkill -f 'vllm_kestrel --stream-test'; sleep 2
rm -f /tmp/q3cmp.log
ls -l /root/kv/models/gguf/*Q4_0.gguf
