import torch
import numpy as np
from PIL import Image
import cv2
import torch
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from depth_anything.dpt import DPT_DINOv2
import os
import torch.nn.functional as F
depth_anything = DPT_DINOv2('vitl', features=256, out_channels= [256, 512, 1024, 1024])
ckpt = torch.load('models/checkpoints/depth_anything_vitl14.pth')
depth_anything.load_state_dict(ckpt)


transform = Compose([
        Resize(
            width=266,
            height=266,
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
input_name = os.listdir('/data2/yifan/mjt/ssart/content')
for i in input_name:
    if int(i[:-4]) <= 20:
        image = Image.open('/data2/yifan/mjt/ssart/content/' + i)
        image = image.convert("RGB")
        image = np.array(image) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        with torch.no_grad():
            depth = depth_anything(image)

        # depth = F.interpolate(depth[None], (266, 266), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        # depth = F.interpolate(depth[None],(266,266),mode='bilinear', align_corners=False)[0, 0]
        depth = depth.cpu().numpy().astype(np.uint8)
        depth = depth[0]
        print(type(depth))
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join('/data2/yifan/mjt/ssart/anything-depth', i,),depth)




