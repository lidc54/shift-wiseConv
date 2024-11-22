import os
from tensorboardX import SummaryWriter

checkpoints_path='checkpoints'
tensorboards_path='tensorboards'
os.makedirs(tensorboards_path,exist_ok=True)
os.makedirs(f'{tensorboards_path}/res',exist_ok=True)
Infinity=0
NaN=0

for exp in os.listdir(checkpoints_path):
    logfile = os.path.join(checkpoints_path, exp, 'log.txt')
    if not os.path.exists(logfile): continue
    tbfile = f'{tensorboards_path}/res/{exp}'
    if os.path.exists(tbfile): 
        continue
    writer = SummaryWriter(tbfile)
    with open(logfile)as f:
        lines=f.readlines()
        ii=0
        for l in lines:
            ii+=1
            try:
                l=eval(l.strip())
                epoch=l['epoch']
            except:
                print()
            for k in l:
                writer.add_scalar(f'SW/{k}',l[k],epoch)
                    
print('tensorboard --logdir res/ --port=8008  --host=172.31.59.66')            
