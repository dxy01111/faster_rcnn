# 读取voc 数据集

import os

import logging
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np

import torch

logging.basicConfig(level=logging.DEBUG)
logger_name = "dataset"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# create formatter
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)


# create file handler
log_path = r"J:\dxy\my_code\faster_rcnn\log\data.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# create stream handler
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class preprocess():
    def __init__(self, fps, min_size, max_size, shape=None, ):
        """
        Parameters
        ----------
        data : [img_1, img_2, ..., img_n], img.shape = (3, nx, ny)
        min_size, max_size: 缩放的最小 最大尺寸，缩放不改变长宽比
        shape : 将图片填充为shape 尺寸的新图片，多余部分填充为 0
        Returns
        -------
        None.

        """
        self.min_size_target, self.max_size_target = min_size, max_size
        self.fps = fps
        if shape is None:
            shape = (max_size, max_size)
        self.shape_new = shape

    def fill(self, img, shape_new):
        if shape_new is None:
            shape_new = (3, self.max_size, self.max_size)
        img_new = torch.zeros(size=(3, shape_new[0], shape_new[1]))
        nx, ny = img.shape[-2:]
        img_new[:, :nx, :ny] = img
        return img_new

    def norm(self, img):
        mean = torch.mean(img)
        std = torch.std(img)
        return img - mean / std

    def resize(self, img):
        shape = torch.tensor(img.shape[-2:])
        min_size, max_size = torch.min(shape), torch.max(shape)
        scalor1 = self.max_size_target / max_size
        scalor2 = self.min_size_target / min_size
        scalor = min([scalor1, scalor2, ])
        # 插值到新的尺寸，使用binlinear 插值
        img_new = torch.nn.functional.interpolate(
            img[None], scale_factor=scalor, mode="bilinear", recompute_scale_factor=True,
            align_corners=False, )
        return img_new

    def forward(self, ):
        res = {}
        i = 0
        for k, fp in self.fps.items():
            i += 1
            print(i)
            r = {}
            img = cv.imread(fp)

            img = torch.Tensor(img)
            img = torch.transpose(img, 0, 2)
            r['init'] = img
            img = self.resize(img)
            img = self.norm(img)
            img = self.fill(img, self.shape_new)
            r['trans'] = img
            res[k] = r
        self.res = r


class Datasets():
    def __init__(self, root_dir,  year="2012", fn_txt="train.txt", ):
        """
        读取 voc 数据集
        ----------
        year : 数据集年份
            DESCRIPTION. The default is "2012".
        vset_name : train，test，valid
            DESCRIPTION. The default is "train".
        root_dir : voc 数据的存储路径
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # 文件路径
        self.root_dir = os.path.join(root_dir, "VOCdevkit", "VOC"+year)

        self.root_img = os.path.join(self.root_dir, "JPEGImages")

        self.root_Annotations = os.path.join(self.root_dir, "Annotations")

        self.root_txt = os.path.join(self.root_dir, "ImageSets", "Action")

        f_txt = os.path.join(self.root_txt, fn_txt)

        with open(f_txt) as f:
            xml_list = [os.path.join(self.root_Annotations, i.strip() + ".xml")
                        for i in f.readlines() if len(i.strip()) > 0]

        # 读取图片配置文件
        img_anno = {}
        for xml_path in xml_list:
            if not os.path.isfile(xml_path):
                logger.WARNING('lose anno file {}'.format(xml_path))
                continue
            else:
                # 读取xml 信息
                with open(xml_path) as f:
                    xml = f.read()
                xml = ET.fromstring(xml)
                res = self.parse_xml(xml)['annotation']
                img_anno[res['filename']] = res

        # 图片路径list
        self.img_fp = {i: self.get_img_path(
            i) for i in img_anno.keys() if self.get_img_path(i) is not None}
        self.img_anno = {k: v for k, v in img_anno.items(
        ) if self.get_img_path(k) is not None}

    def open_img(self, keys: list):
        imgs = {}
        for key in keys:
            fp = self.img_fp[key]
            img = cv.imread(fp)
            imgs[key] = img
        return imgs

    def get_img_path(self, fname):
        fp = os.path.join(self.root_img, fname)
        if os.path.isfile(fp):
            return fp
        else:
            logging.warning("loss img file: fname")

    def __len__(self):
        return self.img_fp.__len__()

    def parse_xml(self, xml):
        res = {}
        for child in xml:
            if "\n" not in child.text:
                res[child.tag] = child.text
            else:
                child_res = self.parse_xml(child)
                if child.tag == 'object':
                    if child.tag not in res.keys():
                        res[child.tag] = [child_res[child.tag]]
                    else:
                        i = res[child.tag]
                        i.append(child_res)
                        res[child.tag] = i
                else:
                    res[child.tag] = child_res[child.tag]
        return {xml.tag: res}


if __name__ == "__main__":
    ds = Datasets(r"J:\dxy\data\VOCtrainval_11-May-2012", )
    pp = preprocess(ds.img_fp, max_size=150, min_size=75, )
    r = pp.forward()
