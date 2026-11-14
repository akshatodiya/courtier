#!/usr/bin/env python3
"""Plot the development-only 8B figure from pinned results and an epoch-4 receipt."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path('~/courtier')
OR = ROOT/'aya_expanse_8b/evals/epoch_study/orbench_run/report/summary.json'
SYC = ROOT/'courtier_sycophancy/analysis_haiku/rates.csv'
REPAIR = ROOT/'courtier_repair/results/RESULTS.md'
HASHES = {OR:'a71ed1dec4dda66a97af66dbbd7ba0b1e1d4d784dee69d02f90b0390c2be860e',
          SYC:'274f0f54fd2424d9f3103bc5fcff7c6b9ae91b383ec4580a36e959d2a04ef84c',
          REPAIR:'033b47ad9c63eece54e5227d4f8e13651ac0207251d7a6d4a8631ff787886cc0'}
EPOCHS = (0,1,3,4,5,7,9)
ALPHAS = ('0','0.25','0.5','0.75','1')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def count(k,n,total):
    if type(k) is not int or type(n) is not int or type(total) is not int or not 0 <= k <= n <= total or n == 0:
        raise ValueError('invalid count / graded / total')
    return (k,n)

def epoch4(path):
    # Receipt is created only after independent validation of the Haiku evidence.
    x = json.loads(path.read_text())
    if (x.get('condition'),x.get('variant'),x.get('metric'),x.get('total')) != ('8b_epoch4','wrong','wrong_adoption',200):
        raise ValueError('invalid epoch-4 receipt identity')
    for prefix in ('judge_complete','generation_complete'):
        p=Path(x[prefix+'_path'])
        h=x[prefix+'_sha256']
        if not p.is_absolute() or not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) or digest(p)!=h:
            raise ValueError(prefix+' receipt hash mismatch')
    # This count receipt is an input from the independent validation step; the
    # receipt hashes are provenance checks, not a replacement for that step.
    return count(x['successes'],x['graded'],x['total'])

def wilson(k,n):
    z=1.959963984540054
    p=k/n
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*p,100*(center-radius),100*(center+radius))

def read_sources(e4):
    for path,expected in HASHES.items():
        if digest(path)!=expected: raise ValueError('source hash changed: '+str(path))
    rows=json.loads(OR.read_text())['results']
    harm=[]
    for e in EPOCHS:
        found=[r for r in rows if r['epoch']==e and r['metric']=='harmful_compliance' and r['judge']=='haiku-5.5']
        if len(found)!=1: raise ValueError('OR epoch coverage mismatch: '+str(e))
        r=found[0]; n=r['completed']; k=round(r['mean']*n/100)
        if r['planned']!=400 or abs(r['mean']-100*k/n)>1e-8: raise ValueError('invalid OR rate')
        harm.append(count(k,n,r['planned']))
    with SYC.open(newline='') as f: rows=list(csv.DictReader(f))
    wrong=[]
    for e in EPOCHS:
        if e==4: wrong.append(e4); continue
        cond='base' if e==0 else 'epoch'+str(e)
        found=[r for r in rows if r['family']=='8b' and r['condition']==cond and r['variant']=='wrong' and r['metric']=='suggested_wrong']
        if len(found)!=1: raise ValueError('sycophancy epoch coverage mismatch: '+str(e))
        r=found[0]; k,n,total=map(int,(r['successes'],r['graded'],r['total']))
        if total!=200 or abs(float(r['rate_pct'])-100*k/n)>1e-8: raise ValueError('invalid sycophancy rate')
        wrong.append(count(k,n,total))
    # The pinned RESULTS table is produced by the existing validators. Parse its
    # integer counts rather than rounded percentages; no local judge is run.
    table={}
    for line in REPAIR.read_text().splitlines():
        if not line.startswith('| '): continue
        cells=[v.strip() for v in line.strip('|').split('|')]
        a=cells[0].split(':')[0].strip()
        if a not in ALPHAS: continue
        if a in table or len(cells)!=5: raise ValueError('invalid repair table')
        nums=[]
        for c in cells[1:4]:
            m=re.fullmatch(r'([\d.]+)% \((\d+)/(\d+)\)',c)
            if not m: raise ValueError('invalid repair cell: '+c)
            pct,k,n=m.groups(); k=int(k); n=int(n)
            if abs(float(pct)-100*k/n)>0.051: raise ValueError('inconsistent repair rate')
            nums.append(count(k,n,400 if len(nums)==0 else 200))
        table[a]=nums
    if set(table)!=set(ALPHAS): raise ValueError('repair alpha coverage mismatch')
    return harm,wrong,[[table[a][i] for a in ALPHAS] for i in range(3)]

def plot(metric_path,out):
    e4=epoch4(metric_path)  # fail before creating output files
    harm,wrong,repair=read_sources(e4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(14,8),sharey=True)
    colors=('#c14b42','#728255','#326a9c')
    def series(ax,x,values,label,color):
        v=[wilson(k,n) for k,n in values]
        y=[x[0] for x in v]
        ax.errorbar(x,y,yerr=[[m-l for m,l,h in v],[h-m for m,l,h in v]],label=label,color=color,
                    marker='o',capsize=3,linewidth=2,markersize=5)
    series(axs[0],EPOCHS,harm,'Harmful compliance',colors[0])
    series(axs[0],EPOCHS,wrong,'Wrong-suggestion adoption',colors[2])
    axs[0].set_xticks(EPOCHS,['Base','1','3','4','5','7','9'])
    axs[0].set_xlabel('Training checkpoint (epoch)')
    axs[0].set_title('Training trajectory',loc='left',fontweight='bold')
    axs[0].set_ylabel('Responses (%)')
    xs=[float(a) for a in ALPHAS]
    for values,label,color in zip(repair,('Harmful compliance','Benign refusal','Wrong-suggestion adoption'),colors):
        series(axs[1],xs,values,label,color)
    axs[1].set_xticks(xs,ALPHAS)
    axs[1].set_xlabel('SafeTune alpha toward base (0 = epoch 7; 1 = base)')
    axs[1].set_title('Repair of epoch 7',loc='left',fontweight='bold')
    for ax in axs:
        ax.set_ylim(0,100); ax.grid(axis='y',alpha=.22); ax.legend(loc='upper right',fontsize=8,frameon=False)
        ax.spines[['top','right']].set_visible(False)
    ns=lambda values: '/'.join(str(n) for _,n in values)
    caption=(f'Development cases; one Haiku judge; bars show 95% Wilson intervals; ungraded responses excluded.\n'
       f'Left graded n (base, 1, 3, 4, 5, 7, 9): harmful {ns(harm)}; wrong suggestion {ns(wrong)}.\n'
       f'Right graded n (alpha 0, 0.25, 0.5, 0.75, 1): harmful {ns(repair[0])}; benign {ns(repair[1])}; wrong suggestion {ns(repair[2])}.\n'
       'Epoch 4 was judged separately after the main analysis; run and batching effects are possible. OR epochs 2, 6 and 8 omitted to align checkpoints.\n'
       'SafeTune WiseFT was applied to LoRA-factor proxies; courtier skill was not evaluated for the repairs.')
    fig.text(.06,.08,caption,fontsize=8,va='bottom',linespacing=1.6)
    fig.suptitle('8B courtier: training and SafeTune repair',x=.06,ha='left',fontweight='bold',fontsize=15)
    fig.subplots_adjust(left=.06,right=.98,top=.88,bottom=.30,wspace=.18)
    outputs=[Path(str(out)+'.png'),Path(str(out)+'.pdf')]
    if any(p.exists() for p in outputs): raise FileExistsError('output already exists; choose a new prefix')
    for p in outputs: fig.savefig(p,dpi=220)
    print('Wrote '+', '.join(map(str,outputs)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch4-metric',required=True,type=Path)
    p.add_argument('--out-prefix',required=True,type=Path)
    a=p.parse_args()
    plot(a.epoch4_metric,a.out_prefix)
