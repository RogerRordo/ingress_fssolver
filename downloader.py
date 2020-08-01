# -*- coding: utf-8 -*-
# 从OBJ_JSON读取portal列表，下载图片到OBJ_DIR，文件以经纬度命名

import requests
import json
from time import sleep

OBJ_JSON = 'ingress_portals.json'
OBJ_DIR = '.\\obj\\'
TIME_GAP = 1  # 间隔1秒下载


def download_img(img_url, lat, lng, cnt, total):
    fname = str(lat) + '_' + str(lng) + '.jpg'
    print('Downloading ' + str(cnt) + '/' + str(total) + ': ' + fname)
    header = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"
    }
    r = requests.get(img_url, headers=header, stream=True)
    if r.status_code == 200:
        f = open(OBJ_DIR + fname, 'wb')
        f.write(r.content)
        f.close()
        del r
        return False
    del r
    return True


def main():
    f = open(OBJ_JSON, 'r')
    pos = json.load(f)['RECORDS']
    f.close()
    total = len(pos)
    fail_list = []
    for i in range(total):
        if pos[i]['url'] == '':
            fail_list.append((pos[i]['lat'], pos[i]['lon']))
            continue
        while download_img(pos[i]['url'], int(float(pos[i]['lat']) * 1E6), int(float(pos[i]['lon']) * 1E6), i + 1,
                           total):
            sleep(TIME_GAP)
        sleep(TIME_GAP)

    print("Fail: ", fail_list)


if __name__ == '__main__':
    main()
