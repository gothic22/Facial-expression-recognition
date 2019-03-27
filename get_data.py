import requests
import os
import json
import cropping
 
# 爬取百度图片，解析页面的函数
def getManyPages(keyword, pages):
    '''
    参数keyword：要下载的影像关键词
    参数pages：需要下载的页面数
    '''
    params = []
 
    for i in range(30, 30 * pages + 30, 30):
        params.append({
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'queryWord': keyword,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': -1,
            'z': '',
            'ic': 0,
            'word': keyword,
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': 0,
            'istype': 2,
            'qc': '',
            'nc': 1,
            'fr': '',
            'pn': i,
            'rn': 30,
            'gsm': '1e',
            '1488942260214': ''
        })
    url = 'https://image.baidu.com/search/acjson'
    #url ='https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1553340006382_R&pv=&ic=0&nc=1&z=0&hd=0&latest=0&copyright=0&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E6%98%8E%E6%98%9F%E8%90%BD%E6%B3%AA'
    urls = []
    for i in params:
        try:
            urls.append(requests.get(url, params=i).json().get('data'))
        except json.decoder.JSONDecodeError:
            print("解析出错")
    return urls
 
# 下载图片并保存
def getImg(dataList, localPath):
    '''
    参数datallist：下载图片的地址集
    参数localPath：保存下载图片的路径
    '''
    if not os.path.exists(localPath):  # 判断是否存在保存路径，如果不存在就创建
        os.mkdir(localPath)
    x = 0
    for list in dataList:
        for i in list:
            if i.get('thumbURL') != None:
                print('正在下载：%s' % i.get('thumbURL'))
                ir = requests.get(i.get('thumbURL'))
                open(localPath + '%d.jpg' % x, 'wb').write(ir.content)
                x += 1
            else:
                print('图片链接不存在')
 
# 根据关键词来下载图片
if __name__ == '__main__':
    dataList = getManyPages('强忍泪水', 50)     # 参数1:关键字，参数2:要下载的页数
    save_path = 'G:/dataset/125/sad1/'
    getImg(dataList,save_path)  
    cropping.clip_image("G:/dataset/125/sad1/","G:/dataset/125/Sad/")    
