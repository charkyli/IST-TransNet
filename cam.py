from utils.gradcam import GradCAM,show_cam_on_image
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.autograd import Variable
def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Evaluation of networks')

    #
    # Dataset parameters
    #
    parser.add_argument('--pkl-path', type=str, default='./result/sirstaug/2022-05-19_14-33-47_agpcnet_1/checkpoint/Iter-13800_mIoU-0.7425_fmeasure-0.8522.pkl',
                        help='checkpoint path')
    parser.add_argument('--image-path', type=str, default=r'./data/picture/30.bmp', help='image path')
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')

    args = parser.parse_args()
    return args
def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input
if __name__ == '__main__':
    args = parse_args()

    # set device
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    print('...load checkpoint: %s' % args.pkl_path)
    net = torch.load(args.pkl_path, map_location='cpu')
    #print(net)
    target_layers = [net.cbam3.sa]
    img_path = args.image_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (args.base_size, args.base_size))) / 255

    # [C, H, W]
    img_tensor = preprocess_image(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    #input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=img_tensor,target_category=0)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()

