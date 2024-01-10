from tensorboardX import SummaryWriter
import os

os.makedirs('tensorboards', exist_ok=True)
writer = SummaryWriter('tensorboards/')

for dir in os.listdir('checkpoints'):
    logtxt=os.path.join('checkpoints',dir,'log.txt')
    if not os.path.exists(logtxt):continue
    with open(logtxt)as f:
        for info in f.readlines():
            try:
                info=eval(info)
                epoch=info['epoch']
                for k in info:
                    if k=='epoch' :continue
                    writer.add_scalar(f'{dir}/{k}', info[k], epoch)
            except:
                pass
writer.close()
print('tensorboard --logdir tensorboards/ --host=172.31.59.66')  
