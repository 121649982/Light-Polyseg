import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
from torchviz import make_dot
import cv2
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

cuda=True
# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    predict_np[predict_np>0.5]=255
    im = Image.fromarray(predict_np).convert('L')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.jpg')
# def save_output(image_name,pred,d_dir):
#
#     predict = pred
#     predict = predict.squeeze()
#     predict_np = predict.cpu().data.numpy()
#
#     im = Image.fromarray(predict_np*255).convert('RGB')
#     img_name = image_name.split(os.sep)[-1]
#     image = io.imread(image_name)
#     imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
#
#     pb_np = np.array(imo)
#
#     aaa = img_name.split(".")
#     bbb = aaa[0:-1]
#     imidx = bbb[0]
#     for i in range(1,len(bbb)):
#         imidx = imidx + "." + bbb[i]
#
#     imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    image_dir='G:/gc/U-2-Net/test_data/segtest/'
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    prediction_dir='G:/gc/U-2-Net/test_data/seg1/'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    model_dir='./saved_models/u2net/u2net_bce_itr_14000_train_0.350883_tar_0.035700.pth'
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(144),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if cuda:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
   #     img=cv2.imread('G:\\gc\\U-2-Net\\test_data\\seg\\cju160wshltz10993i1gmqxbe.jpg')
        inputs_test = inputs_test.type(torch.FloatTensor)

        if cuda:
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

   #     torch.onnx.export(net, inputs_test, "u2net.onnx", verbose=True, input_names=['input1'], output_names=['output1'],opset_version=9)
      #  inputs_test = Variable(inputs_test)

        # net_plot = make_dot(net(inputs_test), params=dict(net.named_parameters()))
        # net_plot.view()

        import time
        t1=time.time()*1000
        d0,d1,d2,d3,d4,d5,d6= net(inputs_test)


        t2 = time.time() * 1000
        print('elapse:', (t2 - t1))


        # normalization
        pred = d0[:,0,:,:]
        pred = normPRED(pred)

        # w=img.shape[1]
        # h=img.shape[0]

    #    img =torch.asarray(img)
     #   img = np.transpose(img, axes=(1, 2, 0))

        # predict = pred
        # predict = predict.squeeze()
        # predict_np = predict.cpu().data.numpy()
        # predict_np=predict_np*125
        # predict_np =cv2.resize(predict_np,(w,h))
        # for ii in range(h):
        #     for jj in range(w):
        #         if predict_np[ii,jj]>20:
        #             img[ii,jj,0] =  img[ii,jj,0]
        #             img[ii, jj, 1] =  img[ii,jj,1]
        #             img[ii, jj, 2] = predict_np[ii,jj]+img[ii, jj, 2]/2
        #
        # cv2.imshow('xx',img)
        # cv2.waitKey(0)
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d0,d1,d2,d3,d4,d5,d6

if __name__ == "__main__":
    main()
