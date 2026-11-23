"""Run in an isolated board workdir: python3 wevis_hot_cat.py DEPLOY BINARY LABEL."""
import base64
import bisect
import collections
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root, binary = map(Path, sys.argv[1:3]); label = sys.argv[3]
image = (root / 'kitten_224.png').read_bytes()
url = 'data:image/png;base64,' + base64.b64encode(image).decode()
body = json.dumps(dict(model='WeVisOne-VL-2B',temperature=0,max_tokens=20,messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=url)),dict(type='text',text='Locate the cat and give its bounding box.')])])).encode()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
endpoint = f'http://127.0.0.1:{port}'
env = dict(os.environ, OMP_NUM_THREADS=os.getenv('OMP_NUM_THREADS', '8'))
with open(label+'.log','w') as log:
    proc = subprocess.Popen([binary,'--serve','--auto-load','--port',str(port),'--model',root/'model-q4-fixed.vqf','--model-id','WeVisOne-VL-2B'],stdout=log,stderr=log,env=env)
    try:
        for _ in range(120):
            if proc.poll() is not None: raise RuntimeError('server exited')
            try:
                with urllib.request.urlopen(endpoint+'/health',timeout=1): break
            except OSError: time.sleep(.25)
        request = urllib.request.Request(endpoint+'/v1/chat/completions',body,{'Content-Type':'application/json'})
        start = time.monotonic()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(lambda: json.load(urllib.request.urlopen(request,timeout=900)))
            if label=='profile':
                time.sleep(2)
                raw=subprocess.check_output(['./pcsamp',str(proc.pid),'180','30000'],text=True)
                Path('pcs.txt').write_text(raw)
                symbols=[]
                for line in subprocess.check_output(['nm','-nS',binary],text=True).splitlines():
                    fields=line.split()
                    if len(fields)==4 and fields[2] in ('t','T'): symbols.append((int(fields[0],16),int(fields[1],16),fields[3]))
                starts=[s[0] for s in symbols]; counts=collections.Counter()
                for line in raw.splitlines()[1:]:
                    count,module,offset=line.split(); pc=int(offset,16)
                    if module!=str(binary): continue
                    a,n,name=symbols[bisect.bisect_right(starts,pc)-1]
                    if a<=pc<a+n: counts[name]+=int(count)
                print(raw.splitlines()[0],sum(counts.values()),counts.most_common(12))
            result=future.result()
        result.update(wall_s=time.monotonic()-start,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),request_sha256=hashlib.sha256(body).hexdigest())
        Path(label+'.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(result['choices'],result['metrics'])
    finally:
        proc.terminate()
        try: proc.wait(5)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
