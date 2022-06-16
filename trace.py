import logging
import tkinter as tk
import tkinter.filedialog
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.draw as draw
import torch
from PIL import Image, ImageTk
from mmcv import VideoReader

from baidu import BaiduAPI

# 配置logger
logger = logging.getLogger('watermark-tracer')
plt.rcParams['font.family'] = 'SimHei'


def detect_watermark_from_video_result(frames, res, threshold=0.1):
    res: pd.DataFrame = res.sort_values(by='confidence', ascending=False)
    frames_np = [np.array(i) for i in frames]
    # 提取最高置信度
    res = res[res['confidence'] > threshold]
    print("检测结果：\n", res)
    w1, h1, w2, h2 = [int(i) for i in res.loc[0].to_list()[:4]]
    wms = [i[h1:h2, w1:w2] for i in frames_np]  # watermarks
    # 增强水印
    wm = estimate_watermark_from_images(wms)
    return wm, [(w1, h1, w2, h2), ]


def detect_watermark_from_img_result(img, res, err_ratio=0.05, threshold=0.1):
    res: pd.DataFrame = res.sort_values(by='confidence', ascending=False)
    img_np = np.array(img)
    # 以最高置信度为主，假如有其他大小相当的检测框则合并
    width, height = None, None
    for i, box in res.iterrows():
        w, h = box['xmax'] - box['xmin'], box['ymax'] - box['ymin']
        if width is None:  # first run
            width, height = w, h
            continue
        if w > width * (1 + err_ratio) or w < width * (1 - err_ratio) \
                or h > height * (1 + err_ratio) or h < height * (1 - err_ratio):
            res.loc[i, 'class'] = 1
        if box['confidence'] < threshold:
            res.loc[i, 'class'] = 1
    res = res.drop(index=res[res['class'] == 1].index)
    print("检测结果：\n", res)
    boxes = [list(map(int, i[1:5])) for i in res.itertuples()]
    # 假如少于等于3个，直接返回，否则根据多幅图像提取水印
    if len(res) <= 3:
        w1, h1, w2, h2 = boxes[0]
        return img_np[h1:h2, w1:w2], boxes
    else:
        # 把所有子图都resize到相同大小
        wms = []  # watermarks
        for w1, h1, w2, h2 in boxes:
            i = img_np[h1:h2, w1:w2]
            i = Image.fromarray(i).resize((int(width), int(height)))
            wms.append(np.array(i))
        # 增强水印
        wm = estimate_watermark_from_images(wms)
        return wm, boxes


def estimate_watermark_from_images(imgs: list, enhance: int = 50):
    # 估计水印
    grad_x = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3), imgs))
    grad_y = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3), imgs))
    Wm_x = np.median(np.array(grad_x), axis=0)
    Wm_y = np.median(np.array(grad_y), axis=0)
    est = poisson_reconstruct(Wm_x, Wm_y)
    # 转换成255的
    est: np.ndarray = 255 * (est - np.min(est)) / (np.max(est) - np.min(est))
    est = est.astype(np.uint8)
    # 寻找增强区域的模版
    channels = []
    for i in range(est.shape[-1]):
        # 二值化
        blur = cv2.GaussianBlur(est[:, :, i], (5, 5), 0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        channels.append(th)
    mask = np.zeros_like(channels[0]).astype(bool)
    for c in channels:
        mask = mask | c.astype(bool)
    mask = mask[:, :, np.newaxis].repeat(3, axis=2)
    # print(mask.shape, est.shape)
    # print(mask.dtype, est.dtype)
    # plt.figure(2)
    # plt.subplot(211)
    # plt.imshow(mask.astype(int)*255)
    # plt.subplot(212)
    # plt.imshow(est)
    # plt.show()
    # 增强
    est = est + enhance * mask
    est: np.ndarray = 255 * (est - np.min(est)) / (np.max(est) - np.min(est))
    est = est.astype(np.uint8)
    return est


def poisson_reconstruct(gradx, grady, kernel_size=3, num_iters=100, h=0.1,
                        boundary_image=None, boundary_zero=True):
    """
    Iterative algorithm for Poisson reconstruction.
    Given the gradx and grady values, find laplacian, and solve for images
    Also return the squared difference of every step.
    h = convergence rate
    """
    fxx = cv2.Sobel(gradx, cv2.CV_64F, 1, 0, ksize=kernel_size)
    fyy = cv2.Sobel(grady, cv2.CV_64F, 0, 1, ksize=kernel_size)
    laplacian = fxx + fyy
    m, n, p = laplacian.shape

    if boundary_zero is True:
        est = np.zeros(laplacian.shape)
    else:
        assert (boundary_image is not None)
        assert (boundary_image.shape == laplacian.shape)
        est = boundary_image.copy()

    est[1:-1, 1:-1, :] = np.random.random((m - 2, n - 2, p))
    loss = []

    for i in range(num_iters):
        old_est = est.copy()
        est[1:-1, 1:-1, :] = 0.25 * (
                est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] +
                est[2:, 1:-1, :] + est[1:-1, 2:, :] -
                h * h * laplacian[1:-1, 1:-1, :]
        )
        error = np.sum(np.square(est - old_est))
        loss.append(error)

    return est


# GUI选择图片/视频
root = tk.Tk()  # 创建一个Tkinter.Tk()实例
file_path = tkinter.filedialog.askopenfilename(
    multiple=False,
    filetypes=[('图片', '.jpg'), ('图片', '.png'), ('视频', '.mp4')]
)
# root.withdraw()  # 将Tkinter.Tk()实例隐藏
file_path = Path(file_path)
if file_path.is_file() is False:
    logger.error("未选择资源")
    exit(0)

# 提取
if file_path.suffix in ['.jpg', '.png', '.jpeg']:  # 图片
    file_type = 'image'
    imgs = [Image.open(file_path).convert('RGB')]
else:  # 视频（平均抽取10帧）
    file_type = 'video'
    video = VideoReader(str(file_path))
    indices = np.linspace(2, video.frame_cnt - 2, 10).astype(int)
    imgs = [Image.fromarray(video.get_frame(i)[:, :, ::-1]) for i in indices]


# 加载模型
logger.info("开始加载YoloV5模型")
model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', source='local')
model = model.cpu()
matplotlib.use('Qt5Agg')
logger.info("YoloV5加载成功")

# 检测
logger.info("检测中")
results = model(imgs)

# 提取水印
results = results.pandas().xyxy
if file_type == 'image':
    if len(results[0]) == 0:
        logger.error("Yolo检测失败")
        exit(0)
    test_wm, box = detect_watermark_from_img_result(imgs[0], results[0])
elif file_type == 'video':
    idx = -1
    for i, result_item in enumerate(results):
        if len(result_item) != 0:
            idx = i
            break
    if idx == -1:
        logger.error("Yolo检测失败")
        print(results)
        exit(0)
    test_wm, box = detect_watermark_from_video_result(imgs, results[idx])
else:
    raise ValueError

# 溯源
api = BaiduAPI('baidu_cfg.json')
mark_res = api.detect_mark(Image.fromarray(test_wm))
if mark_res['result_num'] > 0 and mark_res['result'][0]['probability'] >= 0.7:
    search_res = api.search(mark_res['result'][0]['name'])
    output = f"检测到可能的水印来源：{mark_res['result'][0]['name']}\n" + \
             f"以下是详细信息：\n{search_res[0]['title']} \n{search_res[0]['href']} \n{search_res[0]['summary']}\n" + \
             "获取方式：百度Logo识别+搜索引擎"
    print(output)
    # print(f"检测到可能的水印来源：{mark_res['result'][0]['name']}")
    # print(f"以下是详细信息：\n\t{search_res[0]['title']} \n\t{search_res[0]['href']} \n\t{search_res[0]['summary']}")
    # print("获取方式：百度Logo识别+搜索引擎")
else:
    ocr_res = api.detect_text(Image.fromarray(test_wm))
    if ocr_res['words_result_num'] >= 1:
        ocr_words: str = ocr_res['words_result'][0]['words']
        if '@' in ocr_words:
            ocr_words = ocr_words[ocr_words.index('@'):]
        search_res = api.search(ocr_words)
        # print(f"检测到可能的水印来源：{ocr_words}")
        # print(f"以下是详细信息：\n\t{search_res[0]['title']} \n\t{search_res[0]['href']} \n\t{search_res[0]['summary']}")
        # print("获取方式：百度OCR+搜索引擎")
        output = f"检测到可能的水印来源：{ocr_words}\n" + \
                 f"以下是详细信息：\n{search_res[0]['title']} \n{search_res[0]['href']} \n{search_res[0]['summary']}\n" + \
                 "获取方式：百度OCR+搜索引擎"
        print(output)
    else:
        output = "溯源失败"

# 展示
new_img = np.array(imgs[0])
for point in box:
    rr, cc = draw.rectangle_perimeter(point[:2], end=point[2:], shape=imgs[0].size)
    new_img[cc, rr] = (0, 255, 255)
new_img = Image.fromarray(new_img)
# plt.figure(1)
#
# plt.subplot(121)
# plt.imshow(new_img)
#
# plt.subplot(222)
# plt.imshow(test_wm)
#
# plt.subplot(224)
# ax = plt.gca()
# text_box = TextBox(ax, "结果：", initial=output)
#
# plt.show()

# 展示2
# draw_win(root, new_img, Image.fromarray(test_wm), output)
frame_l = tk.Frame(master=root, relief=tk.RAISED, borderwidth=1)
frame_l.grid(row=0, column=0)
_new_img = new_img.resize((int(new_img.width / new_img.height * 400), 400))
_source_photo = ImageTk.PhotoImage(_new_img)
label1 = tk.Label(master=frame_l, image=_source_photo)
label1.pack()

frame_r = tk.Frame(master=root, relief=tk.RAISED, borderwidth=1)
frame_r.grid(row=0, column=1)
_test_wm = Image.fromarray(test_wm)
_photo = ImageTk.PhotoImage(_test_wm)
label2 = tk.Label(master=frame_r, image=_photo, bg='gray')
label2.grid(row=0, column=0)
label3 = tk.Label(master=frame_r, text=output, justify=tk.LEFT, wraplength=300)  # , width=30
label3.grid(row=1, column=0)
root.mainloop()
