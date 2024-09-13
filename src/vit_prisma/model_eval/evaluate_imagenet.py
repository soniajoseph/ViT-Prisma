
import logging
import torch

import open_clip

from torch.utils.data import DataLoader

from tqdm import tqdm

# import autocast
from torch.cuda.amp import autocast



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, device='cuda'):
    # autocast = get_autocast(args.precision)
    
    # input_dtype = get_input_dtype(args.precision)

    model.to(device)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                output = model(images)
                image_features = output['image_features'] if isinstance(output, dict) else output
                image_features = image_features.cpu()
                logits = 100. * image_features @ classifier
                logits = logits.to(device)
                
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, model_name, pretrained_classifier, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        print('No imagenet data found.')
        return {}
    # if args.zeroshot_frequency == 0:
    #     return {}
    # if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
    #     return {}
    # if args.distributed and not args.horovod:
    #     model = model.module

    logging.info('Starting zero-shot imagenet.')

    if pretrained_classifier is None:
        logging.info('Building zero-shot classifier')
        autocast = get_autocast()
        if tokenizer is None:
            tokenizer = open_clip.get_tokenizer(model_name)
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device='cuda',
                use_tqdm=True,
            )
        logging.info('Built classifier')
    else:
        print("Using pretrained classifier")
        classifier = pretrained_classifier
        del pretrained_classifier

    results = {}
    if 'imagenet-val' in data:
        dataloader = DataLoader(data['imagenet-val'], batch_size=128, num_workers=8, pin_memory=True)
        top1, top5 = run(model, classifier, dataloader)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5


    logging.info('Finished zero-shot imagenet.')

    return results