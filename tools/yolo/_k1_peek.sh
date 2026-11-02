for f in /tmp/q3col_sw_*.log; do
  echo "$f YES=$(grep -c 'Needle retrieved  YES' $f) ALL=$(grep -c 'Needle retrieved' $f)"
done
tail -4 /tmp/q3col.log
uptime
