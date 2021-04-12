import numpy as np
import SimpleITK as sitk
import random
import torch
import os
from decimal import Decimal, ROUND_HALF_UP
import re
import requests
from secret import token

def setSeed(seed: int = 42) -> None:
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def isMasked(img_or_array):
    if isinstance(img_or_array, sitk.Image):
        array = sitk.GetArrayFromImage(img_or_array)

    elif isinstance(img_or_array, np.ndarray):
        array = img_or_array

    if (array > 0).any():
        return True
    else:
        return False


def sendToLineNotify(message):
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {'Authorization': f'Bearer ' + token}
    data = {"message" : message}
    res = requests.post(line_notify_api, headers=headers, data=data)
    if res.status_code == 200:
        print("Suceeded in sending to Line Notify")
    else:
        print("Failed to send")

def getSizeFromString(string, digit=3, link="-"):
    matchobj = re.match(("([0-9]+)" + link) * (digit - 1) + "([0-9]+)", string)
    if matchobj is None:
        print("[ERROR] Invalid size : {}.".format(string))

    size = np.array([int(s) for s in matchobj.groups()])
    return size

def rounding(number, digit):
    """
    This function rounds number at digit.
    number : float
    digit : 0.1 or 0.01 ... max 1
    if digit equals 1, return int, else float.

    """
    x = Decimal(str(number)).quantize(Decimal(str(digit)), ROUND_HALF_UP)
    if str(digit) == "1":
        x = int(x)
    else:
        x = float(x)

    return x

def getImageWithMeta(imageArray, refImage, spacing=None, origin=None, direction=None):
    image = sitk.GetImageFromArray(imageArray)
    if spacing is None:
        spacing = refImage.GetSpacing()
    if origin is None:
        origin = refImage.GetOrigin()
    if direction is None:
        direction = refImage.GetDirection()

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image
