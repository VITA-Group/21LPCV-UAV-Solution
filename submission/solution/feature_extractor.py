import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

class Extractor(object):
        
    def __init__(self, model_path):
        self.net = torch.jit.load(model_path, map_location="cpu").eval()  # change to torchscript model path

        self.norm = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((256, 128), interpolation=3),
            # transforms.Resize((128, 64), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _preprocess(self, im_crops):
        im_batch = torch.cat([self.norm(im).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to('cpu')
            features = self.net(im_batch)
            fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
            features = features.div(fnorm.expand_as(features))
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

