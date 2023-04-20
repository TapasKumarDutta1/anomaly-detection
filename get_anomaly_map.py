
import seaborn as sns
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import glob
import numpy as np
import cv2

from anomaly_detection import STPM

from torchvision import transforms


def get_anomaly_map(path, checkpoint, stage):

    if stage == "stage-1-no-GAN":
        intensity = 0.005
        size = 50
    elif stage == "stage-1-GAN":
        intensity = 0.012
        size = 50
    else:
        intensity = 0.007
        size = 300
    model = STPM(path).load_from_checkpoint(checkpoint)
    for img_path in glob.glob(path+"/*"):
        img = Image.open(img_path).convert("RGB")

        # preprocess image before feeding to model
        data_transforms = transforms.Compose(
            [
                transforms.Resize((768, 768), Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # feed to model to calculate anomaly map
        x = data_transforms(img)
        x = x.view(1, 3, 768, 768)
        features_t, features_s = model.forward(x)
        anomaly_map, a_map_list = model.cal_anomaly_map(
            features_s, features_t, out_size=768
        )

        ret, thresh = cv2.threshold(anomaly_map, intensity, 1, 0)
        contours, hierarchy = cv2.findContours(np.array(thresh, np.uint8), 1, 2)
        areas = []
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, ((768, 768)))
        if (len(areas) > 0) and (max(areas) > size):
            anomaly_map = (anomaly_map > intensity).astype(int)
            anomaly_map = 255 * np.expand_dims(anomaly_map.astype(int), -1)
            pad = np.zeros_like(anomaly_map)
            anomaly_map = [pad, anomaly_map, pad]
            anomaly_map = np.concatenate(anomaly_map, -1)
            anomaly_map = anomaly_map + image
            anomaly_map = anomaly_map / np.max(anomaly_map)
            plt.imshow(anomaly_map)
            plt.show()
        else:
            # plot image
            plt.imshow(image)
            plt.show()
