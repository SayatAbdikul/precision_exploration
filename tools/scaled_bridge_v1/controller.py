"""One controller, immutable attempt accounting, scoped inherited pool leases."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from .common import *
from tools.phase3.worker_locks import exclusive,pool_locks

LIMIT=14400.

def ledger():
    attempts=[]
    for path in sorted((BASE/'budget/attempts').glob('*.json')):
        a=unseal(path);done=path.with_name(path.stem+'.complete')
        finish=unseal(done) if done.exists() else None
        if a['ceiling']!=LIMIT:raise ValueError('new worker ceiling drift')
        charge=finish['charged_seconds'] if finish else a['reserved_seconds']
        if charge<0 or charge>a['reserved_seconds']+.01:raise ValueError('invalid worker accounting')
        attempts.append({**a,'completion':finish,'charged_seconds':charge})
    spent=sum(a['charged_seconds'] for a in attempts)
    return {'ceiling_seconds':LIMIT,'charged_seconds':spent,'remaining_seconds':max(0.,LIMIT-spent),'attempts':attempts}

def dispatch(arguments,reservation,fds):
    reservation=float(math.ceil(reservation));snapshot=ledger()
    if reservation>snapshot['remaining_seconds']:raise BudgetStop(f'cannot reserve {reservation:.0f}s with {snapshot["remaining_seconds"]:.1f}s remaining')
    identity=uuid.uuid4().hex;path=BASE/'budget/attempts'/f'{identity}.json'
    started=time.time();deadline=started+reservation-2.
    immutable(path,{'id':identity,'arguments':arguments,'sources':sources(),'started_epoch':started,
                   'deadline_epoch':deadline,'reserved_seconds':reservation,'ceiling':LIMIT})
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4',
         'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','SCALED_BRIDGE_ATTEMPT':str(path),
         'SCALED_BRIDGE_CPU_FD':str(fds['cpu']),'SCALED_BRIDGE_CUDA_FD':str(fds['cuda'])}
    log=BASE/'budget/logs'/f'{identity}.log';log.parent.mkdir(parents=True,exist_ok=True)
    process=None;state='failed';code=None
    print(f'launch {" ".join(arguments)}; reserve {reservation:.0f}s; remaining {snapshot["remaining_seconds"]:.0f}s',flush=True)
    try:
        with log.open('wb') as stream:
            process=subprocess.Popen([sys.executable,'-m','tools.run.scaled_bridge','worker',*arguments],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=tuple(fds.values()),start_new_session=True)
            code=process.wait(timeout=max(.1,deadline-time.time()))
        state='complete' if code==0 else 'failed'
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        state='interrupted'
        raise
    finally:
        elapsed=time.time()-started
        # On abnormal termination retain the full reservation. A still-running
        # orphan is also bounded by the child's kernel SIGALRM deadline.
        charge=min(reservation,elapsed) if state=='complete' else reservation
        immutable(path.with_name(path.stem+'.complete'),{'status':state,'returncode':code,
                  'elapsed_seconds':elapsed,'charged_seconds':charge,'log':reference(log) if log.exists() else None})
        print(f'{state}: {elapsed:.1f}s wall, {charge:.1f}s charged',flush=True)
    if code!=0:
        print(log.read_text()[-8000:],flush=True)
        raise RuntimeError(f'worker failed; retained attempt {identity}')
    print(log.read_text()[-1800:],flush=True)

class BudgetStop(RuntimeError):pass

def validate_lease():
    import fcntl
    attempt=unseal(Path(os.environ['SCALED_BRIDGE_ATTEMPT']))
    if attempt['sources']!=sources():raise ValueError('worker source identity changed')
    for kind,filename in (('CPU','cpu-native-worker.lock'),('CUDA','native-worker.lock')):
        fd=int(os.environ['SCALED_BRIDGE_'+kind+'_FD']);actual=os.fstat(fd)
        expected=(ROOT/'artifacts/phase3/locks'/filename).stat()
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise ValueError('invalid resource lease')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remaining=attempt['deadline_epoch']-time.time()
    if remaining<=0:raise ValueError('expired experiment reservation')
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,remaining)

def worker(args):
    validate_lease();frozen=sources();action=args[0]
    if action=='prepare':
        from .export import export_all
        export_all()
    elif action=='check':
        from .conformance import primitive_checks
        primitive_checks()
    elif action=='replay':
        from .replay import replay
        replay(args[1],int(args[2]))
    elif action=='panel':
        from .worker import panel
        panel(args[1],args[2],args[3],int(args[4]))
    elif action=='gate':
        from .worker import gate
        gate(args[1])
    else:raise ValueError('unknown bounded worker action')
    if sources()!=frozen:raise ValueError('worker source changed during experiment')

def records(name,mode,backend):
    paths=sorted((run_root()/name/f'{mode}-{backend}').glob('*.json'))
    result=[unseal(p) for p in paths]
    if [r['index'] for r in result]!=list(range(len(result))):raise ValueError('noncontiguous checkpoints')
    return result

def panel_reservation(name,mode,backend,target):
    rows=records(name,mode,backend);count=max(0,target-len(rows))
    if not count:return 0.
    if not rows:return 600. if backend=='cpp' else 240.
    # Upper observed image cost, with first-image oracle included, is
    # conservative for subsequent images; preparation allowance is separate.
    rate=max(r['timing']['execution']+r['preprocess_seconds'] for r in rows)
    return math.ceil(1.5*count*rate+max(60.,1.5*max(r['setup_seconds'] for r in rows))+5.)

def ensure_panel(name,mode,backend,target,fds):
    cost=panel_reservation(name,mode,backend,target)
    if cost:dispatch(['panel',name,mode,backend,str(target)],cost,fds)

def ensure_replay(name,target,fds):
    paths=sorted((run_root()/name/'B').glob('*.json'))
    if len(paths)>=target:return
    if not paths:reserve=600.
    else:
        rate=max(unseal(p)['execution_batch_seconds']+unseal(p)['preparation_batch_seconds'] for p in paths)
        reserve=1.5*math.ceil((target-len(paths))/8)*rate+120.
    dispatch(['replay',name,str(target)],reserve,fds)

def run(check_only=False):
    # Compilation has no inference and is explicitly outside experiment time.
    from .native import build
    build('cpp');build('cuda')
    BASE.mkdir(parents=True,exist_ok=True)
    with exclusive(BASE/'controller.lock'),pool_locks(ROOT) as fds:
        if not (run_root()/'conformance.json').exists():dispatch(['check'],600.,fds)
        if check_only:return
        if not enrollment_path().exists():dispatch(['prepare'],900.,fds)
        # Complete all probes/gates and all paired32 cases before choosing128.
        for name in FORMATS:
            ensure_replay(name,8,fds)
            for mode in ('wide','control'):
                for backend in ('cpp','cuda'):ensure_panel(name,mode,backend,1,fds)
            for mode in ('wide','control'):
                for backend in ('cpp','cuda'):ensure_panel(name,mode,backend,8,fds)
            if not (run_root()/name/'gate.json').exists():dispatch(['gate',name],60.,fds)
        for name in FORMATS:
            ensure_replay(name,32,fds)
            for mode in ('wide','control'):ensure_panel(name,mode,'cuda',32,fds)
        for name in FORMATS:
            from .export import load_export
            ex,_,_=load_export(name);rows=records(name,'wide','cuda')[:32]
            wide=100*sum(r['top5'][0]==int(r['sample']['label']) for r in rows)/32
            anchor=100*sum(r['top5'][0]==int(r['sample']['label']) for r in ex['retained_B_prefix'][:32])/32
            decision={'format':name,'panel':32,'wide_top1':wide,'B_top1':anchor,'quality_pass':wide>=40 and wide>=anchor-20,
                      'rule':PROTOCOL['promotion'],'sources':sources()}
            immutable(run_root()/name/'promotion.json',decision)
            if not decision['quality_pass']:continue
            needed=sum(panel_reservation(name,m,'cuda',128) for m in ('wide','control'))
            paths=sorted((run_root()/name/'B').glob('*.json'))
            if len(paths)<128:
                rate=max(unseal(p)['execution_batch_seconds']+unseal(p)['preparation_batch_seconds'] for p in paths)
                needed+=math.ceil(1.5*math.ceil((128-len(paths))/8)*rate+120)
            if needed>ledger()['remaining_seconds']:
                print(f'{name}:128 deferred; paired reservation {needed:.0f}s exceeds balance',flush=True);continue
            ensure_replay(name,128,fds)
            for mode in ('wide','control'):ensure_panel(name,mode,'cuda',128,fds)
    from .report import write_report
    write_report()
