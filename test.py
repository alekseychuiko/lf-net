# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:04:21 2019

@author: achuiko
"""

import os
import sys
import argparse
import pickle
import videostreamer
import cv2
import tensorflow as tf
import build_network
import time
import numpy as np
import model
import detector

MODEL_PATH = './models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

def convertToKeyPonts(pts, size, ori):
  keypoints = list()
  for i in range(len(pts)):
    keypoints.append(cv2.KeyPoint(pts[i][0], pts[i][1], size[i]*10.0, ori[i]))
  return keypoints

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('--object_path', type=str,
        help='Path to object to detect')
    parser.add_argument('--H', type=int, default=720,
        help='Input image height (default: 720).')
    parser.add_argument('--W', type=int, default=1280,
        help='Input image width (default:1280).')
    parser.add_argument('--camid', type=int, default=0,
        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--show_keypoints', type=int, default=0,
        help='0 - dont show keypoints, 1 - show matched keypoints, 2 - show all keypoints (default: not show)')
    parser.add_argument('--matcher_multiplier', type=float, default=0.8,
        help='Filter matches using the Lowes ratio test (default: 0.8).')
    parser.add_argument('--norm_type', type=int, default=1,
        help='0 - L1, 1 - L2, 2 - L2SQR, 3 - HAMMING, 4 - HAMMING (default: 1)')
    parser.add_argument('--method', type=int, default=0,
        help='0 - RANSAK, 1 - LMEDS, 2 - RHO (default: 0)')
    parser.add_argument('--repr_threshold', type=int, default=3,
        help='Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC and RHO methods only) (default: 3)')
    parser.add_argument('--max_iter', type=int, default=2000,
        help='Maximum number of RANSAC iterations (default: 2000)')
    parser.add_argument('--confidence', type=float, default=0.995,
        help='homography confidence level (default: 0.995).')
    parser.add_argument('--model', type=str, default='./release/models/outdoor/',
                            help='model file or directory')
    parser.add_argument('--top_k', type=int, default=500,
                            help='number of keypoints')

    tmp_config = parser.parse_args()
    print(tmp_config)

    # restore other hyperparams to build model
    if os.path.isdir(tmp_config.model):
        config_path = os.path.join(tmp_config.model, 'config.pkl')
    else:
        config_path = os.path.join(os.path.dirname(tmp_config.model), 'config.pkl')
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except:
        raise ValueError('Fail to open {}'.format(config_path))

    for attr, dst_val in sorted(vars(tmp_config).items()):
        if hasattr(config, attr):
            src_val = getattr(config, attr)
            if src_val != dst_val:
                setattr(config, attr, dst_val)
        else:
            setattr(config, attr, dst_val)
            
    norm_type = cv2.NORM_L1
    if tmp_config.norm_type == 0 : norm_type = cv2.NORM_L1
    elif tmp_config.norm_type == 1 : norm_type = cv2.NORM_L2
    elif tmp_config.norm_type == 2 : norm_type = cv2.NORM_L2SQR
    elif tmp_config.norm_type == 3 : norm_type = cv2.NORM_HAMMING
    else : norm_type = cv2.NORM_HAMMING2
      
    method = cv2.RANSAC
    if tmp_config.method == 0 : method = cv2.RANSAC
    elif tmp_config.method == 1 : method = cv2.LMEDS
    else : method = cv2.RHO
    # This class helps load input images from different sources.
    vs = videostreamer.VideoStreamer("camera", tmp_config.camid, tmp_config.H, tmp_config.W, 1, '')
    
    
    print("Build Networks")
    tf.reset_default_graph()

    photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
    is_training = tf.constant(False) # Always False in testing

    ops = build_network.build_networks(config, photo_ph, is_training)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True 
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # load model
    saver = tf.train.Saver()
    print('Load trained models...')

    if os.path.isdir(config.model):
        checkpoint = tf.train.latest_checkpoint(config.model)
        model_dir = config.model
    else:
        checkpoint = config.model
        model_dir = os.path.dirname(config.model)


    if checkpoint is not None:
        print('Checkpoint', os.path.basename(checkpoint))
        print("[{}] Resuming...".format(time.asctime()))
        saver.restore(sess, checkpoint)
    else:
        raise ValueError('Cannot load model from {}'.format(model_dir))    
    print('Done.')
    
    objDetector = detector.Detector(tmp_config.matcher_multiplier, norm_type, method, tmp_config.repr_threshold, tmp_config.max_iter, tmp_config.confidence)

    win = 'LF-NET Tracker'
    objwin = 'Object'
    cv2.namedWindow(win)
    cv2.namedWindow(objwin)
    
    print('Running Demo.')
  
    obj = model.ModelFile(tmp_config.object_path)
    greyObj = cv2.cvtColor(obj.image, cv2.COLOR_BGR2GRAY)
    
    feed_dict = {
        photo_ph: greyObj[None,...,None].astype(np.float32) / 255.0,
    }
    
    fetch_dict = {
                'kpts': ops['kpts'],
                'feats': ops['feats'],
                'kpts_scale': ops['kpts_scale'],
                'kpts_ori': ops['kpts_ori'],
                'scale_maps': ops['scale_maps'],
                'degree_maps': ops['degree_maps'],
                 }
    outsObj = sess.run(fetch_dict, feed_dict=feed_dict)
    objKeyPoints = convertToKeyPonts(outsObj['kpts'], outsObj['kpts_scale'], outsObj['kpts_ori'])
    if tmp_config.show_keypoints != 0:
        objImg = cv2.drawKeypoints(greyObj, objKeyPoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(objwin, objImg)
    else:
        cv2.imshow(objwin, greyObj)  
        
    while True:

        start = time.time()

        # Get a new image.
        img, status = vs.next_frame()
    
        if status is False:
            break

        # Get points and descriptors.
        start1 = time.time()
        feed_dict = {
            photo_ph:  img[None,...,None].astype(np.float32) / 255.0,
        }
        fetch_dict = {
                'kpts': ops['kpts'],
                'feats': ops['feats'],
                'kpts_scale': ops['kpts_scale'],
                'kpts_ori': ops['kpts_ori'],
                'scale_maps': ops['scale_maps'],
                'degree_maps': ops['degree_maps'],
                 }
        outs = sess.run(fetch_dict, feed_dict=feed_dict)
        imgKeyPoints = convertToKeyPonts(outs['kpts'], outs['kpts_scale'], outs['kpts_ori'])
        out = objDetector.detect((np.dstack((img, img, img))).astype('uint8'), 
                       objKeyPoints, imgKeyPoints, outsObj['feats'], outs['feats'], obj, tmp_config.show_keypoints)
        
        end1 = time.time()
        cv2.imshow(win, out)
        
        key = cv2.waitKey(tmp_config.waitkey) & 0xFF
        if key == ord('q'):
            print('Quitting, \'q\' pressed.')
            break
    
        end = time.time()
        net_t = (1./ float(end1 - start))
        total_t = (1./ float(end - start))
        print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' % (vs.i, net_t, total_t))
        
    cv2.destroyAllWindows()

    print('==> Finshed Demo.')