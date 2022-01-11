import argparse
import os
import os.path as osp
import time
from torch.backends import cudnn
import torch
from torch import optim
from torchvision import transforms

import trainer
from DataLoaderX import DataLoaderX
from data_prefetcher import data_prefetcher
from datasets import get_dataloader
from models.loss import DetectionCriterion
from models.model import DetectionModel
from torchvision.models import vgg16

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("traindata", default="data/WIDER/wider_face_split/wider_face_train_bbx_gt.txt")
    parser.add_argument("valdata", default="data/WIDER/wider_face_split/wider_face_val_bbx_gt.txt")
    parser.add_argument("--dataset-root", default="data/WIDER")
    parser.add_argument("--dataset", default="WIDERFace")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_edge", default=False)

    return parser.parse_args()


def main():
    args = arguments()
    num_templates = 25  # aka the number of clusters

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_loader, _ = get_dataloader(args.traindata, args, num_templates,
                                     img_transforms=img_transforms, enable_edge=args.enable_edge)
    # prefetcher = data_prefetcher(train_loader)

    model = DetectionModel(num_objects=1, num_templates=num_templates, enable_edge=args.enable_edge)
    loss_fn = DetectionCriterion(num_templates)

    # directory where we'll store model weights
    weights_dir = "weights"
    if not osp.exists(weights_dir):
        os.mkdir(weights_dir)

    # check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    optimizer = optim.SGD(model.learnable_parameters(args.lr), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=20,
                                          last_epoch=args.start_epoch-1)

    # train and evalute for `epochs`
    for epoch in range(args.start_epoch, args.epochs):
        trainer.train(model, loss_fn, optimizer, train_loader, epoch, device=device)
        scheduler.step()

        if (epoch+1) % args.save_every == 0:
            file_name = ''
            if args.enable_edge:
                file_name = "edge_"
            trainer.save_checkpoint({
                'epoch': epoch + 1,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename=file_name + "checkpoint_{0}.pth".format(epoch+1), save_path=weights_dir)


if __name__ == '__main__':
    main()
