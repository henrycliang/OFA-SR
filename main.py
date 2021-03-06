import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from trainer_distill import Trainer_Distill
from trainer_distill2 import Trainer_Distill2

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)     ###setting the log and the train information

if checkpoint.ok:
    loader = data.Data(args)                ##data loader
    model = model.Model(args, checkpoint)
    print('The current model is:',args.model)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    if args.distill_model=='trainer_distill':
        t = Trainer_Distill(args, loader, model, loss, checkpoint) # Use the Trainer_Distill version
    elif args.distill_model == 'trainer_distill2':
        t = Trainer_Distill2(args, loader, model, loss, checkpoint)
    else:
        t = Trainer(args, loader, model, loss, checkpoint)  # Don't use the Trainer_Distill version
    while not t.terminate():

        t.train()
        t.test()

    checkpoint.done()

