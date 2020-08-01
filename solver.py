# -*- coding: utf-8 -*-
# 从OBJ_DIR读取po图，从SUB_DIR读取题目子图，计算哈希完成匹配并画图

import photohash
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

OBJ_DIR = '.\\obj\\'
SUB_DIR = '.\\sub\\'
INTEL_LINKS = 'intel_links.txt'  # 保存link表
DRAW_SAVE = 'draw.jpg'  # 保存画图
CHAR_SIZE = 70  # 画图时一个字符的大小
EDGE_SIZE = 10  # 边距大小


def getObjs():
    res = []
    objFileList = os.listdir(OBJ_DIR)
    for objFile in objFileList:
        objPath = OBJ_DIR + objFile
        objHash = photohash.average_hash(objPath, 16)
        temp2 = os.path.splitext(objFile)[0].split('_')
        temp = {'path': objPath, 'hash': objHash, 'lat': int(temp2[0]), 'lng': int(temp2[1])}
        res.append(temp)
    return res


def getSubs():
    res = []
    subFileList = os.listdir(SUB_DIR)
    for subFile in subFileList:
        subPath = SUB_DIR + subFile
        subHash = photohash.average_hash(subPath, 16)
        temp2 = os.path.splitext(subFile)[0].split('_')
        temp = {'path': subPath, 'hash': subHash, 'col': int(temp2[0]), 'row': int(temp2[1]), 'match_result': None}
        res.append(temp)
    return res


def match(objs, subs):
    for sub in subs:
        mnd = 1000
        for obj in objs:
            d = photohash.hash_distance(sub['hash'], obj['hash'])
            if (d < mnd):
                mnd = d
                sub['match_result'] = obj


def drawComp(sub):
    imgl = cv2.imread(sub['path'], 0)
    imgr = cv2.imread(sub['match_result']['path'], 0)

    plt.subplot(1, 2, 1)
    plt.imshow(imgl, cmap='gray')
    plt.title('sub')

    plt.subplot(1, 2, 2)
    plt.imshow(imgr, cmap='gray')
    plt.title('obj')

    plt.show()


def genIntelLinks(subs):
    res = set()
    for sub in subs:
        lat = str(sub['match_result']['lat'] / 1E6)
        lng = str(sub['match_result']['lng'] / 1E6)
        res.add('https://intel.ingress.com/intel?pll=' + lat + ',' + lng)

    f = open(INTEL_LINKS, 'w')
    for elem in res:
        f.write(elem + '\n')
    f.close()

    return res


def drawChar(subs):
    subs.sort(key=lambda x: (x['col'], x['row']))  # subs排序
    # print(subs)
    colsnum = subs[len(subs) - 1]['col']  # 有几列
    canvas = np.zeros((CHAR_SIZE + EDGE_SIZE * 2, CHAR_SIZE * colsnum + EDGE_SIZE * (colsnum + 1), 3),
                      dtype="uint8")  # 生成画布
    mxlat = [-1 << 32] * (colsnum + 1)
    mxlng = [-1 << 32] * (colsnum + 1)
    mnlat = [1 << 32] * (colsnum + 1)
    mnlng = [1 << 32] * (colsnum + 1)
    for sub in subs:
        mxlat[sub['col']] = max(mxlat[sub['col']], sub['match_result']['lat'])
        mnlat[sub['col']] = min(mnlat[sub['col']], sub['match_result']['lat'])
        mxlng[sub['col']] = max(mxlng[sub['col']], sub['match_result']['lng'])
        mnlng[sub['col']] = min(mnlng[sub['col']], sub['match_result']['lng'])
    # print(mxlat, mnlat, mxlng, mnlng)
    for i in range(1, len(subs)):
        st = subs[i - 1]
        ed = subs[i]
        if (ed['col'] != st['col']):
            continue
        mxlat0, mxlng0, mnlat0, mnlng0 = mxlat[st['col']], mxlng[st['col']], mnlat[st['col']], mnlng[st['col']]
        stlat0, stlng0 = st['match_result']['lat'], st['match_result']['lng']
        edlat0, edlng0 = ed['match_result']['lat'], ed['match_result']['lng']
        # print(mxlat0, mxlng0, mnlat0, mnlng0)
        # print(stlat0, stlng0, edlat0, edlng0)
        stx = int((mxlat0 - stlat0) / (mxlat0 - mnlat0) * CHAR_SIZE) + EDGE_SIZE
        sty = int(
            (stlng0 - mnlng0) / (mxlng0 - mnlng0) * CHAR_SIZE) + CHAR_SIZE * (st['col'] - 1) + EDGE_SIZE * st['col']
        edx = int((mxlat0 - edlat0) / (mxlat0 - mnlat0) * CHAR_SIZE) + EDGE_SIZE
        edy = int(
            (edlng0 - mnlng0) / (mxlng0 - mnlng0) * CHAR_SIZE) + CHAR_SIZE * (st['col'] - 1) + EDGE_SIZE * st['col']
        # print((sty, stx), (edy, edx))
        cv2.line(canvas, (sty, stx), (edy, edx), (255, 255, 255))

    cv2.imwrite(DRAW_SAVE, canvas)
    cv2.imshow('drawChar', canvas)
    cv2.waitKey(0)


def main():
    objs = getObjs()
    # print(objs)

    subs = getSubs()
    # print(subs)

    # 匹配
    match(objs, subs)

    # 手动检查匹配
    # for sub in subs:
    #    drawComp(sub)

    # 生成intel links
    genIntelLinks(subs)

    # 画图
    drawChar(subs)


if __name__ == '__main__':
    main()
