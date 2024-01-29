import torch
import torch.nn as nn
from torchvision import models
import cv2
class Perceptual_loss(nn.Module):
    def __init__(self, loss):
        super(Perceptual_loss, self).__init__()
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model


    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
#fake_B = cv2.imread("/home/zc/Deep leaning/SNCyclegan/results/sn_L1_SSIM_big_aligned_house_cyclegan/test_latest/images/0000_fake_A.png")

#real_B = cv2.imread("/home/zc/Deep leaning/SNCyclegan/results/sn_L1_SSIM_big_aligned_house_cyclegan/test_latest/images/0000_real_A.png")

#content_loss = Perceptual_loss(torch.nn.MSELoss())
#loss_PerceptualLoss = content_loss.get_loss( fake_B, real_B)
#print(Perceptual_loss)

