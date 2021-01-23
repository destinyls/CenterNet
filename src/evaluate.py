from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

from tools.evaluation.kitti_utils.eval import kitti_eval
from tools.evaluation.kitti_utils import kitti_common as kitti

def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    val_mAP = []
    iteration_list = []
    checkpoints_path = opt.save_dir
    for model_name in os.listdir(checkpoints_path):
        if "pth" not in model_name or "final" in model_name:
            continue
        print("model_name: ", model_name)
        iteration = int(model_name.split(".")[0].split('_')[1])
        iteration_list.append(iteration)
    iteration_list = sorted(iteration_list)  
    for iteration in iteration_list:
        model_name = "model_{:07d}.pth".format(iteration)
        opt.load_model = os.path.join(checkpoints_path, model_name)

        Dataset = dataset_factory[opt.dataset]
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
        print(opt)
        Logger(opt)
        Detector = detector_factory[opt.task]
        split = 'val' if not opt.trainval else 'test'
        dataset = Dataset(opt, split)
        detector = Detector(opt)

        results = {}
        num_iters = len(dataset)
        bar = Bar('{}'.format(opt.exp_id), max=num_iters)
        time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
        avg_time_stats = {t: AverageMeter() for t in time_stats}
        for ind in range(num_iters):
            img_id = dataset.images[ind]
            img_info = dataset.coco.loadImgs(ids=[img_id])[0]
            img_path = os.path.join(dataset.img_dir, img_info['file_name'])

            if opt.task == 'ddd':
                ret = detector.run(img_path, img_info['calib'])
            else:
                ret = detector.run(img_path)
            results[img_id] = ret['results']
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                        ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
            for t in avg_time_stats:
                avg_time_stats[t].update(ret[t])
                Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
            bar.next()
        bar.finish()
        dataset.run_eval(results, opt.save_dir)

        gt_label_path = "/root/CenterNet/data/kitti/training/label_2/"
        pred_label_path = os.path.join(opt.save_dir, 'results')
        pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
        gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
        result, ret_dict = kitti_eval(gt_annos, pred_annos, ["Car", "Pedestrian", "Cyclist"])

        if ret_dict is not None:
            mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_loose"]
            val_mAP.append(mAP_3d_moderate)
            with open(os.path.join(checkpoints_path, "val_mAP.json"),'w') as file_object:
                json.dump(val_mAP, file_object)
            with open(os.path.join(checkpoints_path, 'epoch_result_{:07d}_{}.txt'.format(iteration, round(mAP_3d_moderate, 2))), "w") as f:
                f.write(result)


if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
