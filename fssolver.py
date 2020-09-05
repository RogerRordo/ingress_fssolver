# -*- coding: utf-8 -*-

import os
import cv2
import json
import math
import argparse
import requests
import numpy as np
from time import sleep
from ingressAPI import IntelMap, MapTiles
from PIL import Image, ImageDraw

# Python2 and Python3 compatibility
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser


def create_config(config_path):
    ''' Parse config '''
    config = dict()
    config_raw = ConfigParser()
    config_raw.read(config_path, encoding='utf-8')
    config['cookies'] = config_raw.get('Crawl', 'COOKIES', raw=True)
    config['lat'] = config_raw.getfloat('Crawl', 'LAT')
    config['lng'] = config_raw.getfloat('Crawl', 'LNG')
    config['rad'] = config_raw.getint('Crawl', 'RAD')
    config['poList'] = config_raw.get('Crawl', 'PO_LIST')
    config['picDir'] = config_raw.get('Crawl', 'PICDIR')
    config['delay'] = config_raw.getint('Crawl', 'DELAY')

    config['prob'] = config_raw.get('Solve', 'PROB')
    config['urlList'] = config_raw.get('Solve', 'URL_LIST')
    config['sol'] = config_raw.get('Solve', 'SOL')
    config['initRow'] = config_raw.getint('Solve', 'INIT_ROW')
    config['minPix'] = config_raw.getint('Solve', 'MIN_PIX')
    config['minColWidth'] = config_raw.getint('Solve', 'MIN_COL_WIDTH')
    config['bgColorR'] = config_raw.getint('Solve', 'BGCOLOR_R')
    config['bgColorG'] = config_raw.getint('Solve', 'BGCOLOR_G')
    config['bgColorB'] = config_raw.getint('Solve', 'BGCOLOR_B')
    config['charSize'] = config_raw.getint('Solve', 'CHAR_SIZE')
    config['edgeSize'] = config_raw.getint('Solve', 'EDGE_SIZE')

    return config


def getAllPortals(login, tiles, totalTiles):
    zoom = 15
    timedOutItems = []
    poDetails = []
    poID = []
    tilesData = []
    for idx, tile in enumerate(tiles):
        iitc_xtile = int(tile[0])
        iitc_ytile = int(tile[1])

        iitcTileName = ('{0}_{1}_{2}_0_8_100').format(zoom, iitc_xtile, iitc_ytile)
        currentTile = idx + 1
        print(str('{0}/{1} Getting portals from tile : {2}').format(currentTile, totalTiles, iitcTileName))
        try:
            tilesData.append(login.get_entities([iitcTileName]))
            if config['delay'] > 0:
                sleep(config['delay'])
        except:
            print(str('[!] Something went wrong while getting portal from tile {0}').format(currentTile))
    for tile_data in tilesData:
        try:
            if 'result' in tile_data:
                for data in tile_data['result']['map']:
                    if 'error' in tile_data['result']['map'][data]:
                        timedOutItems.append(data)
                    else:
                        for entry in tile_data['result']['map'][data]['gameEntities']:
                            if entry[2][0] == 'p':
                                poID.append(entry[0])
                                poDetails.append(entry[2])
        except:
            print('[!] Could not parse all prtals')
    return poDetails, poID


def crawler():
    IngressLogin = IntelMap(config['cookies'])

    if IngressLogin.getCookieStatus() is False:
        return

    dpl = 111000  # distance per lat
    dlat = 1.0 * config['rad'] / dpl
    dlng = 1.0 * config['rad'] / (dpl * math.cos(config['lat'] / 180))
    bbox = [config['lng'] - dlng, config['lat'] - dlat, config['lng'] + dlng, config['lat'] + dlat]
    print('BBOX : ', bbox)

    mTiles = MapTiles(bbox)
    tiles = mTiles.getTiles()
    print('Tiles : ', tiles)
    print('Number of tiles in boundry are : ' + str(len(tiles)))

    poDetails, poId = getAllPortals(IngressLogin, tiles, len(tiles))

    poList = []
    for idx, val in enumerate(poId):
        lat = (poDetails[idx][2]) / 1e6
        lng = (poDetails[idx][3]) / 1e6
        url = poDetails[idx][7]
        if url == '':
            print('[!] Picture lost : ' + str(lat) + ',' + str(lng))
            continue
        poList.append({'lat': lat, 'lng': lng, 'url': url})
    print('Total portals : ' + str(len(poList)))
    record = {'RECORDS': poList}
    with open(config['poList'], 'w') as f:
        json.dump(record, f, ensure_ascii=False)
        print('Portals JSON has been saved as : ' + config['poList'])


def download_img(img_url, lat, lng, cnt, total):
    ''' 下载单张图片 '''
    fname = str(lat) + '_' + str(lng) + '.jpg'
    print('Downloading ' + str(cnt) + '/' + str(total) + ' : ' + fname)
    header = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"
    }
    r = requests.get(img_url, headers=header, stream=True)
    if r.status_code == 200:
        f = open(config['picDir'] + '\\' + fname, 'wb')
        f.write(r.content)
        f.close()
        del r
        return False
    del r
    return True


def downloader():
    with open(config['poList'], 'r') as f:
        poJson = json.load(f)
        poList = poJson['RECORDS']
    total = len(poList)
    print(str(total) + ' portals loaded from ' + config['poList'])
    for i in range(total):
        url = poList[i]['url']
        lat = float(poList[i]['lat'])
        lng = float(poList[i]['lng'])
        while download_img(url, int(lat * 1E6), int(lng * 1E6), i + 1, total):
            sleep(config['delay'])
        sleep(config['delay'])


def isSameColor(a, b):
    ''' 两个rgb同色的标准 '''
    return (abs(a[0] - b[0]) <= 10 and abs(a[1] - b[1]) <= 10 and abs(a[2] - b[2]) <= 10)


def drawSubfig(subs):
    ''' 画出子图位置 '''
    targetImage = Image.open(config['prob'])
    targetDraw = ImageDraw.Draw(targetImage)
    for elem in subs:
        [x0, y0, x1, y1] = elem['cord']
        targetDraw.line([(y0, x0), (y0, x1), (y1, x1), (y1, x0), (y0, x0)], fill=(255, 0, 0), width=5)
    targetImage.show()


def splitTarget_flood(sx, sy, p, img):
    ''' flood挖出联通块 '''
    bgcolor = [config['bgColorR'], config['bgColorG'], config['bgColorB']]
    cnt, x0, y0, x1, y1 = 1, sx, sy, sx, sy
    height = img.shape[0]
    weight = img.shape[1]
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
            if (p[xx][yy] or isSameColor(img[xx][yy], bgcolor)):
                continue
            b.append([xx, yy])
            cnt = cnt + 1
            p[xx][yy] = True

            x0 = min(x0, x)
            y0 = min(y0, y)
            x1 = max(x1, x)
            y1 = max(y1, y)

    return [cnt, x0, y0, x1, y1]


def hashDistance(a, b):
    ''' 即有多少字符不同 '''
    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(a, b)))


def ahash(image, hash_size=16):
    ''' 均值hash '''
    image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_CUBIC)  # 缩放
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 灰度图
    gray = [i for j in gray for i in j]  # 压平
    avg = sum(gray) / len(gray)
    bits = ''.join(map(lambda x: '1' if x > avg else '0', gray))
    hashformat = "0{hashlength}x".format(hashlength=hash_size**2 // 4)
    return int(bits, 2).__format__(hashformat)


def spliter():
    print('Splitting ...')
    bgcolor = [config['bgColorR'], config['bgColorG'], config['bgColorB']]
    subs = []
    img = cv2.imread(config['prob'])
    height = img.shape[0]
    weight = img.shape[1]
    p = [[False for j in range(weight)] for i in range(height)]
    col = 0
    lastColY = -config['minColWidth'] - 1
    for y in range(weight):
        for x in range(height):
            if (x >= config['initRow'] and not p[x][y] and not isSameColor(img[x][y], bgcolor)):
                [cnt, x0, y0, x1, y1] = splitTarget_flood(x, y, p, img)
                if cnt <= config['minPix']:
                    continue
                if y > lastColY + config['minColWidth']:
                    col = col + 1
                    lastColY = y
                temp = {'cord': [x0, y0, x1, y1], 'col': col, 'row': 0, 'hash': '', 'match_result': None}
                subs.append(temp)
                # print(temp)

    subs.sort(key=lambda x: (x['col'], x['cord'][0]))  # 排序以获取row
    lastCol = 0
    row = 1
    for sub in subs:
        row = row + 1
        if (sub['col'] != lastCol):
            row = 1
            lastCol = sub['col']
        sub['row'] = row

        [x0, y0, x1, y1] = sub['cord']
        sub['hash'] = ahash(img[x0:x1, y0:y1])

    drawSubfig(subs)

    return subs


def solver(subs, toCheck):
    print('Solving ...')

    # 获取爬取po的hash等信息，存进objs
    objs = []
    objFileList = os.listdir(config['picDir'])
    for objFile in objFileList:
        objPath = config['picDir'] + '\\' + objFile
        img = cv2.imread(objPath)
        objHash = ahash(img)
        temp2 = os.path.splitext(objFile)[0].split('_')
        temp = {'path': objPath, 'hash': objHash, 'lat': int(temp2[0]), 'lng': int(temp2[1])}
        objs.append(temp)

    # 匹配
    for sub in subs:
        mnd = 1000
        for obj in objs:
            d = hashDistance(sub['hash'], obj['hash'])
            if (d < mnd):
                mnd = d
                sub['match_result'] = obj

    # 生成intel links
    urls = set()
    for sub in subs:
        lat = str(sub['match_result']['lat'] / 1E6)
        lng = str(sub['match_result']['lng'] / 1E6)
        urls.add('https://intel.ingress.com/intel?pll=' + lat + ',' + lng)
    with open(config['urlList'], 'w') as f:
        for elem in urls:
            f.write(elem + '\n')

    # 画图
    subs.sort(key=lambda x: (x['col'], x['row']))  # subs排序
    # print(subs)
    colsnum = subs[len(subs) - 1]['col']  # 有几列
    canvas = np.zeros((config['charSize'] + config['edgeSize'] * 2, config['charSize'] * colsnum + config['edgeSize'] *
                       (colsnum + 1), 3),
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
        stx = int((mxlat0 - stlat0) / (mxlat0 - mnlat0) * config['charSize']) + config['edgeSize']
        sty = int((stlng0 - mnlng0) / (mxlng0 - mnlng0) *
                  config['charSize']) + config['charSize'] * (st['col'] - 1) + config['edgeSize'] * st['col']
        edx = int((mxlat0 - edlat0) / (mxlat0 - mnlat0) * config['charSize']) + config['edgeSize']
        edy = int((edlng0 - mnlng0) / (mxlng0 - mnlng0) *
                  config['charSize']) + config['charSize'] * (st['col'] - 1) + config['edgeSize'] * st['col']
        # print((sty, stx), (edy, edx))
        cv2.line(canvas, (sty, stx), (edy, edx), (255, 255, 255))
    cv2.imwrite(config['sol'], canvas)
    cv2.imshow('solution', canvas)
    cv2.waitKey(0)

    # 检查
    if toCheck:
        img = cv2.imread(config['prob'])
        for sub in subs:
            [x0, y0, x1, y1] = sub['cord']
            obj = sub['match_result']
            img1 = cv2.resize(img[x0:x1, y0:y1], (512, 512))
            img2 = cv2.resize(cv2.imread(obj['path']), (512, 512))
            h_all = np.hstack((img1, img2))
            windowName = str(obj['lat']) + '_' + str(obj['lng'])
            cv2.imshow(windowName, h_all)
            cv2.waitKey(0)
            cv2.destroyWindow(windowName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default.ini', help='Config file to use')
    parser.add_argument('-c', '--crawl', action='store_true', help='Crawl portals data')
    parser.add_argument('-d', '--download', action='store_true', help='Download pictures of portals')
    parser.add_argument('-s', '--solve', action='store_true', help='Solve FS code')
    parser.add_argument('--check', action='store_true', help='Check when solving FS code')
    args = parser.parse_args()
    config_path = args.config
    config = create_config(config_path)

    if args.crawl:
        crawler()

    if args.download:
        if not os.path.exists(config['poList']):
            print('[!] ' + config['poList'] + ' not found')
        elif not os.path.exists(config['picDir']):
            os.makedirs(config['picDir'])
            downloader()
        elif not os.path.isdir(config['picDir']):
            print('[!] ' + config['picDir'] + ' is not a directory')
        else:
            downloader()

    if args.solve:
        if not os.path.exists(config['prob']):
            print('[!] ' + config['prob'] + ' not found')
        else:
            solver(spliter(), args.check)
