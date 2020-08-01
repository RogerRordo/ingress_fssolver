# -*- coding: utf-8 -*-
# 从Target分割出子图，保存到SUB_DIR

import cv2
from PIL import Image, ImageDraw

TARGET = '2020_08.jpg'
SUB_DIR = '.\\sub\\'
INIT_ROW = 0  # 80 # 跳过target的前几行（图片总标题）
BGCOLOR = [50, 50, 50]  # 背景色
CNT_LIMIT = 5000  # 至少几个像素算子图
COL_WIDTH_LIMIT = 100  # 至少多少个像素算一子图列


# 两个rgb同色的标准
def isSameColor(a, b):
    return (abs(a[0] - b[0]) <= 5 and abs(a[1] - b[1]) <= 5 and abs(a[2] - b[2]) <= 5)


# flood挖出联通块
def splitTarget_flood(sx, sy, p, img):
    cnt, x0, y0, x1, y1 = 1, sx, sy, sx, sy
    height = img.shape[0]  # 高度
    weight = img.shape[1]  # 宽度
    d = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    b = [[sx, sy]]  # 队列？栈！。。
    p[sx][sy] = True

    while len(b) > 0:
        [x, y] = b.pop()
        # print(x, y, len(b), cnt)
        for dd in d:
            xx, yy = x + dd[0], y + dd[1]  # 转移坐标
            if (xx < 0 or xx >= height or yy < 0 or yy >= weight):  # 越界
                continue
            if (p[xx][yy] or isSameColor(img[xx][yy], BGCOLOR)):
                continue
            b.append([xx, yy])
            cnt = cnt + 1
            p[xx][yy] = True

            if x < x0:
                x0 = x
            if y < y0:
                y0 = y
            if x > x1:
                x1 = x
            if y > y1:
                y1 = y

    return [cnt, x0, y0, x1, y1]


# 画出子图位置
def drawSubfig(subfig):
    targetImage = Image.open(TARGET)
    targetDraw = ImageDraw.Draw(targetImage)
    for elem in subfig:
        [x0, y0, x1, y1] = elem['cord']
        targetDraw.line([(y0, x0), (y0, x1), (y1, x1), (y1, x0), (y0, x0)], fill=(255, 0, 0), width=5)
    targetImage.show()


def main():
    print('Target: ', TARGET)
    print('Splitting target...')
    subfig = []
    img = cv2.imread(TARGET)
    height = img.shape[0]  # 高度
    weight = img.shape[1]  # 宽度
    p = [[False for j in range(weight)] for i in range(height)]
    col = 0
    lastColY = -COL_WIDTH_LIMIT - 1
    for y in range(weight):
        for x in range(height):
            if (x >= INIT_ROW and not p[x][y] and not isSameColor(img[x][y], BGCOLOR)):
                [cnt, x0, y0, x1, y1] = splitTarget_flood(x, y, p, img)
                if cnt <= CNT_LIMIT:
                    continue
                if y > lastColY + COL_WIDTH_LIMIT:
                    col = col + 1
                    lastColY = y
                temp = {'cord': [x0, y0, x1, y1], 'col': col, 'row': 0, 'path': ''}
                subfig.append(temp)
                # print(temp)

    subfig.sort(key=lambda x: (x['col'], x['cord'][0]))  # 排序以获取row
    lastCol = 0
    for elem in subfig:
        if (elem['col'] == lastCol):
            row = row + 1
        else:
            row = 1
            lastCol = elem['col']
        elem['row'] = row
        spath = SUB_DIR + str(elem['col']) + '_' + str(elem['row']) + '.jpg'
        elem['path'] = spath
        [x0, y0, x1, y1] = elem['cord']
        cv2.imwrite(spath, img[x0:x1, y0:y1])
    '''
    targetImage = Image.open(TARGET)
    targetDraw = ImageDraw.Draw(targetImage)
    for y in range(weight):
        for x in range(height):
            if (p[x][y]):
                targetDraw.point((y, x), fill=(255, 0, 0))
    targetImage.show()
    '''

    drawSubfig(subfig)


if __name__ == '__main__':
    main()
