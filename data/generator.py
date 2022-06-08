from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw as draw
import random
from typing import Union


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


def add_watermark(image: Image.Image, watermark: Image.Image,
                  alpha: Union[float, tuple], position: str,
                  scale: Union[float, tuple] = .3,
                  offset_scale: Union[float, tuple] = .02,
                  tile_density: Union[tuple, tuple[float]] = (0.5, 1),
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
                    'center', 'left_sidebar', 'right_sidebar', 'bottom_sidebar']:
        match position:
            case 'l':
                position = offset, ih // 2 - wh // 2
            case 'r':
                position = iw - ww - offset, ih // 2 - wh // 2
            case 't':
                position = iw // 2 - ww // 2, offset
            case 'b':
                position = iw // 2 - ww // 2, ih - wh - offset
            case 'tl':
                position = offset, offset
            case 'tr':
                position = iw - ww - offset, offset
            case 'bl':
                position = offset, ih - wh - offset
            case 'br':
                position = iw - ww - offset, ih - wh - offset
            case 'center':
                position = iw // 2 - ww // 2, ih // 2 - wh // 2
            case 'right_sidebar':
                position = iw - ww, int(ih * 0.75)
            case 'left_sidebar':
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


if __name__ == '__main__':
    water_mark = get_watermark('test_imgs/wechat.png', '@打哈欠的小汪')
    img, boxes = add_watermark(
        Image.open('test_imgs/n01440764_12063.jpg'),
        watermark=water_mark,
        alpha=0.7,
        position='right_sidebar',
        scale=.1,
        tile_density=(.5, 2),
        tile_rotate=-30
    )
    new_img = np.array(img)
    for point in boxes:
        rr, cc = draw.rectangle_perimeter(point[:2], end=point[2:], shape=img.size)
        new_img[cc, rr] = (0, 255, 255, 255)
    new_img = Image.fromarray(new_img)
    plt.imshow(new_img)
    plt.show()