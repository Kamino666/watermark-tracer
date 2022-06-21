import random
from typing import Union
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from os.path import join as pjoin
from itertools import cycle, islice
import os
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw as draw
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm


def rotate_points(points: np.ndarray, matrix: np.ndarray):
    """
    :param points: N, 4, 2
    :param matrix: 2, 3
    :return:
    """
    # build M 3,3
    M = np.eye(3, 3, dtype=float)
    M[0:2, :] = matrix
    # build P N,4,2 -> 4N,2 -> 4N,3 -> 3,4N
    P = points.reshape((-1, 2))
    P = np.concatenate(
        [P, np.ones((P.shape[0], 1), dtype=float)],
        axis=1
    ).transpose((1, 0))
    # do the math 3,3 @ 3,4N = 3,4N
    Y = M @ P
    # recover 3,4N -> 2,4N -> 2,N,4 -> N,4,2
    ys = np.split(Y, 3, axis=0)
    Y = np.stack([ys[0], ys[1]], axis=0).reshape((2, -1, 4)).transpose((1, 2, 0))
    return Y


def crop_points(points: np.ndarray, box: tuple):
    x1, y1, x2, y2 = box
    point_x = (points[:, :, 0] > x1) & (points[:, :, 0] < x2)
    point_y = (points[:, :, 1] > y1) & (points[:, :, 1] < y2)
    point_mask = (point_x & point_y)
    box_mask = point_mask[:, 0] & point_mask[:, 1] & point_mask[:, 2] & point_mask[:, 3]
    return points[box_mask]


def bound_points(points: np.ndarray):
    point_x, point_y = points[:, :, 0], points[:, :, 1]
    # all is N
    min_x, max_x = np.min(point_x, axis=1), np.max(point_x, axis=1)
    min_y, max_y = np.min(point_y, axis=1), np.max(point_y, axis=1)
    # all is N,2
    top_left = np.stack([min_x, min_y], axis=1)
    bottom_right = np.stack([max_x, max_y], axis=1)
    # N,2,2
    return np.concatenate([top_left[:, np.newaxis, :], bottom_right[:, np.newaxis, :]], axis=1)


def determine_random_value(x: Union[int, float, tuple], ranges=(0.0, 1.0)):
    if type(x) is tuple:
        assert len(x) == 2
        assert ranges[0] < x[0] <= ranges[1] and ranges[0] < x[1] <= ranges[1] and x[0] <= x[1]
        return random.uniform(x[0], x[1])
    elif type(x) is float or type(x) is int:
        assert ranges[0] < x <= ranges[1]
        return x
    else:
        raise ValueError


def get_tile_watermark_layer(watermark, tile_density, tile_rotate, iw, ih, ww, wh):
    # 创建水印层
    watermark_layer = Image.new('RGBA', (iw * 2, ih * 2), (0, 0, 0, 0))
    point_list = []
    for i in range(0, iw * 2, int(ww + ww * tile_density[0])):
        for j in range(0, ih * 2, int(wh + wh * tile_density[1])):
            watermark_layer.paste(watermark, (i, j))
            point_list.append([
                (i, j), (i + ww, j), (i, j + wh), (i + ww, j + wh)
            ])
    # 水印层旋转
    watermark_layer = watermark_layer.rotate(tile_rotate)
    rotate_mat = cv2.getRotationMatrix2D((iw, ih), tile_rotate, 1)
    rotated_points = rotate_points(np.array(point_list), rotate_mat)
    # watermark_layer.show()
    # 水印层裁剪
    watermark_layer = watermark_layer.crop((iw // 2, ih // 2, iw // 2 * 3, ih // 2 * 3))
    rotated_points = crop_points(rotated_points, (iw // 2, ih // 2, iw // 2 * 3, ih // 2 * 3))
    rotated_points = bound_points(rotated_points).astype(int)  # N,2,2
    rotated_points[:, :, 0] -= iw // 2
    rotated_points[:, :, 1] -= ih // 2
    return watermark_layer, rotated_points.reshape((-1, 4))


def apply_watermark_to_image(image, watermark, position=(0, 0)):
    width, height = image.size
    new_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    new_img.paste(image, (0, 0))
    new_img.paste(watermark, position, mask=watermark)
    return new_img


# The main function of adding a watermark to the image
def add_watermark(image: Image.Image, watermark: Image.Image,
                  alpha: Union[float, tuple], position: str,
                  scale: Union[float, tuple] = .3,
                  offset_scale: Union[float, tuple] = .02,
                  tile_density: Union[tuple] = (0.5, 1),
                  tile_rotate: Union[float, tuple] = 45):
    """
    :return: image, (x, y, x+w, y+h)
    --->x/w
    ⭣
    y/h
    """
    # 确定随机值
    alpha = determine_random_value(alpha)
    scale = determine_random_value(scale)
    tile_rotate = determine_random_value(tile_rotate, (-90, 90))
    tile_density = (
        determine_random_value(tile_density[0], (0, 10)),
        determine_random_value(tile_density[1], (0, 10))
    )
    offset_scale = determine_random_value(offset_scale)

    # 假如是左右的sidebar，则预处理watermark
    if position in ['left_sidebar', 'right_sidebar']:
        new_watermark = Image.new('RGBA', (int(watermark.width * 1.3), int(watermark.height * 2)), (0, 0, 0, 255))
        center = new_watermark.width // 2, new_watermark.height // 2
        new_watermark.paste(watermark,
                            (center[0] - watermark.width // 2, center[1] - watermark.height // 2),
                            mask=watermark)
        watermark = new_watermark

    # apply alpha
    watermark = np.array(watermark, dtype=float)
    watermark[:, :, 3] *= alpha
    watermark = Image.fromarray(watermark.astype(np.uint8))

    # make scale
    ratio = watermark.width / watermark.height
    new_wh = image.height * scale
    watermark = watermark.resize((int(new_wh * ratio), int(new_wh),), Image.Resampling.LANCZOS)

    # get position: (w, h)
    (iw, ih), (ww, wh) = image.size, watermark.size
    offset = int(min(iw, ih) * offset_scale)
    if position in ['l', 'r', 'b', 't', 'tl', 'tr', 'bl', 'br',
                    'center', 'left_sidebar', 'right_sidebar']:
        if position == 'l':
            position = offset, ih // 2 - wh // 2
        elif position == 'r':
            position = iw - ww - offset, ih // 2 - wh // 2
        elif position == 't':
            position = iw // 2 - ww // 2, offset
        elif position == 'b':
            position = iw // 2 - ww // 2, ih - wh - offset
        elif position == 'tl':
            position = offset, offset
        elif position == 'tr':
            position = iw - ww - offset, offset
        elif position == 'bl':
            position = offset, ih - wh - offset
        elif position == 'br':
            position = iw - ww - offset, ih - wh - offset
        elif position == 'center':
            position = iw // 2 - ww // 2, ih // 2 - wh // 2
        elif position == 'right_sidebar':
            position = iw - ww, int(ih * 0.75)
        elif position == 'left_sidebar':
            position = 0, int(ih * 0.75)
    elif position == 'random':
        position = (
            int(random.uniform(offset, iw - ww - offset)),
            int(random.uniform(offset, ih - wh - offset)),
        )
    elif position == 'tile':
        pass
    else:
        raise ValueError

    # process
    if position == 'tile':
        watermark_layer, rotated_points = get_tile_watermark_layer(
            watermark, tile_density, tile_rotate,
            iw, ih, ww, wh
        )
        # 最终叠加
        new_img = apply_watermark_to_image(image, watermark_layer, (0, 0))
        return new_img, rotated_points.tolist()
    else:
        new_img = apply_watermark_to_image(image, watermark, position)
        return new_img, [(position[0], position[1], position[0] + ww, position[1] + wh)]


# The main function of generating a watermark
def get_watermark(logo, text: str, font_path="./SourceHanSansCN-Regular.otf"):
    logo: Image.Image = Image.open(logo)
    if text == '':
        return logo
    logo_ratio = logo.width / logo.height
    font = ImageFont.truetype(font_path, 256)
    # First. get the bounding box of text
    tmp = Image.new('RGBA', (10, 10))
    draw_tool = ImageDraw.Draw(tmp)
    box = draw_tool.textbbox((0, 0), text, font=font)
    height, width = box[3] + box[1], box[2] - box[0]
    # Second. generate water mark
    # print((width+int(height * logo_ratio * 1.1), height))
    watermark = Image.new('RGBA', (width + int(height * logo_ratio * 1.1), height), (0, 0, 0, 0))
    resized_logo = logo.resize((int(height * logo_ratio), height))
    watermark.paste(resized_logo, (0, 0), mask=resized_logo)
    draw_tool = ImageDraw.Draw(watermark)
    draw_tool.text((int(height * logo_ratio * 1.1), 0), text, font=font, fill=(255, 255, 255),
                   stroke_width=0)  # *1.1 is margin
    # print(draw_tool.textbbox((logo.width, 0), text, font=font))
    return watermark


class WatermarkedImageGenerator:
    def __init__(self, data_dir: str, output_dir: str, num_workers=4, schemata=None, schemata_weight=None,
                 logo_dir="./logos", name_source="./douban_usernames.txt"):
        """
        :param data_dir: scan all images(jpg,png) in the folder/sub-folder
        """
        self.img_list = list(Path(data_dir).rglob('*.jpg')) + \
                        list(Path(data_dir).rglob('*.png')) + \
                        list(Path(data_dir).rglob('*.JPEG'))
        if len(self.img_list) == 0:
            raise ValueError(f"Images are not found in the folder {data_dir}.")
        random.shuffle(self.img_list)
        print(f'reading {len(self.img_list)} images')

        self.schemata = self.get_default_schemata() if schemata is None else schemata
        self.schemata_weight = schemata_weight
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() is False:
            os.mkdir(self.output_dir)
            os.mkdir(pjoin(self.output_dir, 'images'))
            os.mkdir(pjoin(self.output_dir, 'labels'))

        # load logo
        self.combined_logos = [i for i in Path(logo_dir, 'combined').glob("*")]
        self.independent_logos = [i for i in Path(logo_dir, 'independent').glob("*")]
        # with open(pjoin(logo_dir, 'combined.txt'), encoding='utf-8') as f:
        #     self.combined_logos = [pjoin(logo_dir, i + '.png') for i in f.read().split('\n')]
        # with open(pjoin(logo_dir, 'independent.txt'), encoding='utf-8') as f:
        #     self.independent_logos = [pjoin(logo_dir, i + '.png') for i in f.read().split('\n')]
        print(self.combined_logos)
        print(self.independent_logos)
        with open(name_source, encoding='utf-8') as f:
            self.name_list = [i[:-1] for i in f.readlines()]

    @staticmethod
    def get_default_schemata():
        schemata = [
            ('combined', (0.4, 1.0), 'random', (.08, .14), (.02, .03), ((1, 1.5), (.8, 2)), 0),
            ('independent', (0.4, 1.0), 'random', (.08, .14), (.02, .03), ((1, 1.5), (.8, 2)), 0),
            ('combined', (0.4, 1.0), 'tile', (.08, .09), (.02, .03), ((.3, .5), (1.2, 4)), (-45, 45)),
            ('independent', (0.4, 1.0), 'tile', (.08, .11), (.02, .03), ((1, 1.5), (1.2, 4)), (-45, 45)),
            ('combined', (0.4, 1.0), 'sidebar', (.08, .14), (.02, .03), ((1, 1.5), (.8, 2)), 0),
        ]
        return schemata

    def generate(self, num):
        """generate num images with watermark"""
        if num == -1:
            num = len(self.img_list)
        pool = ProcessPoolExecutor(max_workers=self.num_workers)
        futures = []
        for img_path in tqdm(islice(cycle(self.img_list), 0, num), desc='allocating', total=num):
            idx = np.random.choice(range(len(self.schemata)), p=self.schemata_weight)
            futures.append(
                pool.submit(self._gen_wm_imgs, img_path, self.schemata[idx])
            )
        for future in tqdm(futures, desc='processing'):
            img, boxes, path = future.result()
            img.save(pjoin(self.output_dir, 'images', path.stem + '.jpg'))
            with open(pjoin(self.output_dir, 'labels', path.stem + '.txt'), 'w+') as f:
                for box in boxes:
                    content = [
                        0,
                        min((box[2] + box[0]) / 2, img.width) / img.width,
                        min((box[3] + box[1]) / 2, img.height) / img.height,
                        min(box[2] - box[0], img.width) / img.width,
                        min(box[3] - box[1], img.height) / img.width,
                    ]
                    f.write(" ".join(map(lambda x: str(x), content)) + '\n')

    def _gen_wm_imgs(self, img_path, schema):
        if schema[0] == 'independent':
            logo = random.choice(self.independent_logos)
            wm = get_watermark(logo, '')
        else:
            logo = random.choice(self.combined_logos)
            name = random.choice(self.name_list)
            wm = get_watermark(logo, random.choice(['', '@']) + name)

        if schema[2] == 'random':
            position = random.choice(['l', 'r', 'b', 't', 'tl', 'tr', 'bl', 'br', 'center', 'random'])
        elif schema[2] == 'sidebar':
            position = random.choice(['left_sidebar', 'right_sidebar'])
        else:
            position = schema[2]

        img, boxes = add_watermark(
            Image.open(str(img_path)),
            watermark=wm,
            alpha=schema[1],
            position=position,
            scale=schema[3],
            offset_scale=schema[4],
            tile_density=schema[5],
            tile_rotate=schema[6]
        )
        return img.convert('RGB'), boxes, img_path


def parse_args():
    # python generator.py --test
    # python generator.py --images_dir "ImageNet 1000 (mini)" --output_dir "./WatermarkDataset" -n 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help="测试代码能否正常运行")
    parser.add_argument('--images_dir', type=str, default='./ImageNet 1000 (mini)', help="输入图片目录")
    parser.add_argument('--logo_dir', type=str, default='./logos', help="输入水印logo目录")
    parser.add_argument('--output_dir', type=str, default='./WatermarkDataset', help="输出目录")
    parser.add_argument('--num_workers', type=int, default=4, help="进程数")
    parser.add_argument('-n', '--num', type=int, default=-1, help="生成图片数量")
    parser.add_argument('--seed', type=int, default=2022, help="种子")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.test is True:
        water_mark = get_watermark("test_imgs/知乎.png", "@伸懒腰")
        img, boxes = add_watermark(
            Image.open('test_imgs/big_road.jpg'),
            watermark=water_mark,
            alpha=0.3,
            position='tile',
            scale=.07,
            offset_scale=.03,
            tile_density=(0.5, 1.2),
            tile_rotate=10
        )
        img.save('sample6.png')
        new_img = np.array(img)
        for point in boxes:
            rr, cc = draw.rectangle_perimeter(point[:2], end=point[2:], shape=img.size)
            new_img[cc, rr] = (0, 255, 255, 255)
        new_img = Image.fromarray(new_img)
        plt.imshow(new_img)
        plt.show()
    else:
        images_dir = Path(args.images_dir)
        output_dir = Path(args.output_dir)
        assert images_dir.is_dir() and images_dir.exists()
        output_dir.mkdir(exist_ok=True)
        output_sub_image_dir = output_dir / 'images'
        output_lbl_image_dir = output_dir / 'labels'
        output_sub_image_dir.mkdir(exist_ok=True)
        output_lbl_image_dir.mkdir(exist_ok=True)

        random.seed(args.seed)
        generator = WatermarkedImageGenerator(
            args.images_dir, args.output_dir,
            num_workers=args.num_workers,
            schemata_weight=[.3, .3, .1, .1, .2],
            logo_dir=args.logo_dir
        )
        generator.generate(args.num)
