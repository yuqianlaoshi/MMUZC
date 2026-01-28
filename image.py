
import os
import numpy as np
# 读取图片
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import cv2
import pandas as pd
image = cv2.imread('image.jpg')
if image is None:
    print("图片加载失败，请检查路径和文件名")
else:
    print("图片已加载，尺寸：", image.shape)


CSV_PATH = 'sensor04271_converted.csv'
RESOLUTION_X = 0.081329  # 米/像素
RESOLUTION_Y = 0.106019
LON0, LAT0 = 116.303056, 39.9125  # 左下角坐标 (lon, lat)

H, W = image.shape[:2]
def coord2pixel(lon: float, lat: float):
    """
    将地理坐标（十进制经纬度）转换为图像像素坐标（行列）。

    参数
    ----
    lon : float
        经度（十进制）
    lat : float
        纬度（十进制）

    返回
    ----
    (row, col) : tuple(int, int)
        对应的像素行列坐标（OpenCV 读取后图像的 (row, col) 形式）
    """
    # 计算相对于左下角坐标的水平/垂直距离（米）
    delta_lon_m = (lon - LON0) * 111320 * np.cos(np.radians(LAT0))
    delta_lat_m = (lat - LAT0) * 110540

    col = int(round(delta_lon_m / RESOLUTION_X))
    row = int(round(delta_lat_m / RESOLUTION_Y))

    # 注意 OpenCV 图像坐标系：原点在左上角
    row = image.shape[0] - row - 1  # 翻转 y 轴

    return row, col

def crop_patch(lon: float, lat: float, patch_size: int = 64):
    """
    以 (lon, lat) 为中心裁剪 patch_size×patch_size 的遥感块
    返回：
        patch: ndarray, 形状 (patch_size, patch_size, C)
        (r0, c0): 实际裁剪窗口左上角在原图中的坐标，便于后续定位
    若中心点位于图像外则返回 None
    """
    r, c = coord2pixel(lon, lat)
    #姚丹阳我谢谢你
    if not (0 <= r < H and 0 <= c < W):
        print('中心点在图像外')
        c=max(0,c)
        c=min(c,W)
        r=max(0,r)
        r=min(r,H)
        print(r,c)

    half = patch_size // 2
    r0 = max(r - half, 0)
    c0 = max(c - half, 0)
    # 防止越界右/下
    r1 = min(r0 + patch_size, H)
    c1 = min(c0 + patch_size, W)
    # 如果因越界裁剪不足 patch_size，整体平移补齐
    if (r1 - r0) < patch_size:
        r0 = r1 - patch_size
    if (c1 - c0) < patch_size:
        c0 = c1 - patch_size
    r0, c0 = max(r0, 0), max(c0, 0)   # 再次保险

    patch = image[r0:r0+patch_size, c0:c0+patch_size]
    return patch, (r0, c0)

OUT_DIR='rsdata'

def dd2dms(dd: float, precision: int = 4):
    """
    十进制度 → 度分秒
    返回 dict: {'deg': int, 'min': int, 'sec': float}
    """
    sign = -1 if dd < 0 else 1
    dd = abs(dd)

    deg = int(dd)
    rem = (dd - deg) * 60
    min_ = int(rem)
    sec = round((rem - min_) * 60, precision)

    # 处理 60 进位
    if sec >= 60:
        sec -= 60
        min_ += 1
    if min_ >= 60:
        min_ -= 60
        deg += 1

    return {'deg': sign * deg, 'min': min_, 'sec': sec}
# 示例用法
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, usecols=['Lat', 'Lon'])
    cnt=0
    for idx, row in df.iterrows():
        cnt+=1
        lat, lon = float(row['Lat']), float(row['Lon'])
        patch, (r0, c0) = crop_patch(lon, lat,640)
        """
        print(lat,lon)
        print(dd2dms(lat,4),dd2dms(lat,4))
        print(r0,c0)
        """
        if patch is None:
            print(f'跳过越界点: Lat={lat}, Lon={lon}')
            continue

        # 生成文件名：保留 6 位小数
        fname = f'image{cnt}.jpg'
        fpath = os.path.join(OUT_DIR, fname)
        cv2.imwrite(fpath, patch)
