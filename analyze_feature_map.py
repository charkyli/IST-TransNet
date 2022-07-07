from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from argparse import ArgumentParser
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
from models import get_segmentation_model
# #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data_transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
# model_weight_path = "./result/sirstaug/2022-05-02_14-33-07_agpcnet_1/checkpoint/Iter-36770_mIoU-0.7495_fmeasure-0.8568.pkl"
# #model = get_segmentation_model('agpcnet_1')
# #print(model)
# model=torch.load(model_weight_path,map_location=torch.device('cpu'))
# img = Image.open("./data/1.bmp").convert("RGB")
# img = data_transform(img)
# img_show = np.squeeze(img.detach().numpy())
# img_show = np.transpose(img_show,[1,2,0])
# img_show = (img_show * 255).astype(np.uint8)

# img = torch.unsqueeze(img,dim=0)

# output = model(img)
# for idx, feature_map in enumerate(output):
#     #[B,C,H,W]->[C,H,W]
#     im = np.squeeze(feature_map.detach().numpy())
#     #[C,H,W]->[H,W,C]
#     im =  np.transpose(im,[1,2,0])
#     ratio = np.power(2,idx+1)
#     im = cv2.resize(im[:,:,0], dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    
#     label = "conf_mask_stride=" + str(ratio)
#     plt.figure(label)
#     plt.imshow(im,cmap='jet')
#     cb = plt.colorbar()
#     tick_locator = ticker.MaxNLocator(nbins=5)
#     cb.locator = tick_locator
#     cb.update_ticks()

# plt.figure("img_show")
# plt.imshow(img_show)

# plt.show()
def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Inference of AGPCNet')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path', type=str, default='./result/sirstaug/2022-05-04_13-42-27_agpcnet_1/checkpoint/Iter-39350_mIoU-0.7567_fmeasure-0.8615.pkl',
                        help='checkpoint path')

    #
    # Test image parameters
    #
    parser.add_argument('--image-path', type=str, default=r'./data/0.bmp', help='image path')
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

    # load network
    print('...load checkpoint: %s' % args.pkl_path)
    net = torch.load(args.pkl_path, map_location=torch.device('cpu'))
    net.eval()

    # load image
    print('...loading test image: %s' % args.image_path)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (args.base_size, args.base_size)))/255
    input = preprocess_image(img)

    # inference in cpu
    print('...inference in progress')
    with torch.no_grad():
        output = net(input)
        for idx, feature_map in enumerate( output ):
            # [B,C,H,W]->[C,H,W]
            # feature_map = feature_map.cpu().detach().numpy().reshape(256,256)
            # feature_map = feature_map >0
            # plt.figure()
            # plt.subplot(111),plt.imshow(feature_map,cmap='jet')
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig('./cpm_afm/final_layer_feat%d.jpg'%idx)
            im = np.squeeze(feature_map.cpu().numpy())
            
                # [C,H,W]->[H,W,C]
            #im = np.transpose( im, [1, 2, 0] )
            
            #im = im[2,:,:]
           
           ####将每个通道可视化
            # for i in range(128):
            #     plt.figure()
            #     #plt.subplot(8,16,i+1)
            #     plt.imshow(im[:,:,i],cmap='jet')
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.savefig('./vit_all_channel/all_ch%d.png'%i)
            #ratio = 16*np.power(1/8, idx+1)
            ratio = np.power(1/2, idx)
            im = cv2.resize( im, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC )
            im = im >0
            #print(img.shape)
            label = "conf_mask_stride=" #+ str( ratio )
            plt.figure( label )
            plt.imshow(im,cmap='gray')
            cb = plt.colorbar()
            tick_locator = ticker.MaxNLocator( nbins=5 )
            cb.locator = tick_locator
            cb.update_ticks()
            plt.savefig("./vit_all_channel/%d.jpg"%idx)
                
        
            