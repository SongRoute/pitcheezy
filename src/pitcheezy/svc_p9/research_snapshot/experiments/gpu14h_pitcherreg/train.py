"""원005 pitcher T의 학습률/결합 dropout만 바꾸고 원 fit을 직접 재사용한다."""
from experiments.gpu14h_identity import train as identity

PARENT='pitcher_t32'
VARIANTS={f'lr{lrid}_drop{dropid}':{'lr':lr,'dropout':drop} for lrid,lr in [('100',.001),('060',.0006),('030',.0003)] for dropid,drop in [('010',.1),('020',.2),('030',.3)] if (lr,drop)!=(.001,.1)}


def recipe(producer,variant):
    if variant not in VARIANTS:raise ValueError('fixed8 variants; baseline never refit')
    r=identity.recipe(producer,PARENT);v=VARIANTS[variant]
    if (r['base']['training']['lr']!=.001 or r['base']['model']['dropout']!=.1 or r['base']['model']['encoder_dropout']!=.1):raise ValueError('original005 parent recipe')
    r['base']['training']['lr']=v['lr']
    r['base']['model']['dropout']=v['dropout'];r['base']['model']['encoder_dropout']=v['dropout']
    return r


def run(producer,variant,seed,package,out,synthetic=False,device='cuda'):
    return identity.fit(recipe(producer,variant),PARENT,seed,package,out,synthetic=synthetic,device=device)
