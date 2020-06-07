import argparse

import os

import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import gluoncv as gcv

from mxnet import gluon, autograd
from mxnet.gluon import nn

from mxnet.gluon.data.vision import transforms

from gluoncv import model_zoo, utils

from gluoncv.data import batchify

ctx = [mx.cpu()]


def predict_food(im_fname, output_filename, threshold=0.5, print_outputs=False):

    net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)

    base_classes = ['bowl',
                    'cup',
                    'banana',
                    'apple',
                    'sandwich',
                    'orange',
                    'broccoli',
                    'carrot',
                    'hot dog',
                    'pizza',
                    'donut',
                    'cake',
                    'bottle']

    net.reset_class(classes=base_classes, reuse_weights=base_classes)

    params_path = './trained_parameters/'
    symbol_file = os.path.join(params_path, 'ResNet50_v2_epochs50-lr0.001-wd0.001-symbol.json')
    params_file = os.path.join(params_path, 'ResNet50_v2_epochs50-lr0.001-wd0.001-0000.params')
    
    food_classes = ['borscht', 'lagman', 'manty', 'plov', 'samsy']

    all_classes = base_classes + food_classes

    food_net = nn.SymbolBlock.imports(symbol_file, ['data'], params_file, ctx=ctx)

    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    x, orig_img = gcv.data.transforms.presets.yolo.load_test(im_fname)

    box_ids, scores, bboxes = net.forward(x)

    box_ids_np, scores_np, bboxes_np = box_ids[0].asnumpy(), scores[0].asnumpy(), bboxes[0].asnumpy()

    bowl_mask = ((box_ids_np == 0) & (scores_np > threshold))

    bowl_ids = np.where(bowl_mask)

    bowl_boxes = bboxes_np[bowl_mask.ravel(), :]

    if (len(bowl_boxes) > 0):
        bowl_images = [orig_img[int(box[1]): int(box[3]), int(box[0]): int(box[2])] for box in bowl_boxes.tolist()]

        bowl_batch_img = batchify.Stack()([transform_fn(mx.nd.array(img)) for img in bowl_images]).copyto(ctx[0])

        food_outputs = mx.nd.softmax(food_net(bowl_batch_img))

        food_scores = food_outputs.max(axis=1)

        food_labels = food_outputs.argmax(axis=1) + len(base_classes)

        all_classes = base_classes + food_classes

        box_ids = np.delete(box_ids_np, bowl_ids[0], axis=0)
        scores = np.delete(scores_np, bowl_ids[0], axis=0)
        bboxes = np.delete(bboxes_np, bowl_ids[0], axis=0)

        box_ids = np.concatenate((box_ids, food_labels.asnumpy().reshape(-1, 1)), axis=0)
        scores = np.concatenate((scores, food_scores.asnumpy().reshape(-1, 1)), axis=0)
        bboxes = np.concatenate((bboxes, bowl_boxes.reshape(-1, 4)), axis=0)

    else:
        bboxes = bboxes[0].asnumpy()
        scores = scores[0].asnumpy()
        box_ids = box_ids[0].asnumpy()

    if (print_outputs):
        confident_mask = (scores >= threshold)
        confident_classes = box_ids[confident_mask]
        confident_scores = scores[confident_mask]
        confident_boxes = bboxes[confident_mask.ravel()]

        for i in range(len(confident_boxes)):
            print(f"{all_classes[int(confident_classes[i])]:10} \t {confident_scores[i]:.5f}\t{confident_boxes[i]}")
        

    utils.viz.plot_bbox(orig_img,
                        bboxes,
                        scores,
                        box_ids,
                        class_names=all_classes,
                        thresh=threshold)

    # plt.rc('figure', figsize=(20,20))
    plt.savefig(f"./predictions/{output_filename}", dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict food objects in the image")
    parser.add_argument("-f", "--file", help="default flag to read from image file", metavar="<path to file>")
    parser.add_argument("-u", "--url", help="download image fro URL and read it", metavar="<url of image>")
    parser.add_argument("-w", "--write", default="prediction.jpg", help="save image, default='prediction.jpg'", metavar="<filename>")
    parser.add_argument("-t", "--threshold", help="set threshold for prediction score", default=0.5, type=float, metavar="<float number>")
    parser.add_argument("-p", "--print", help="print prediction outputs", action="store_true")

    args = parser.parse_args()

    if args.file:
        predict_food(args.file, output_filename=args.write, threshold=args.threshold, print_outputs=args.print)

    if args.url:
        im_address = args.url
        im_fname = utils.download(im_address, path='image.jpg', overwrite=True)

        predict_food(im_fname, output_filename=args.write, threshold=args.threshold, print_outputs=args.print)
