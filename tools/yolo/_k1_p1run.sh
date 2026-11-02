if [ -f /tmp/p1.pid ] && kill -0 $(cat /tmp/p1.pid) 2>/dev/null; then
  echo RUNNING; cat /proc/loadavg; tail -5 /tmp/p1.log; exit 0
fi
if [ -f /tmp/p1.log ] && grep -q P1_END /tmp/p1.log; then
  cat /tmp/p1.log; exit 0
fi
nohup bash -c 'for m in 0 1 2; do echo "== load $(cat /proc/loadavg)"; OMP_WAIT_POLICY=active /tmp/bime4 $m 5; done; echo P1_END' > /tmp/p1.log 2>&1 &
echo $! > /tmp/p1.pid
echo STARTED
