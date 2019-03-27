from urllib import request,parse
import os
import json
'''
用urllib爬取post请求数据  网址：视觉中国
'''
def path_name(filename):
    #获取当前工作路径
    file_path = str(os.path.dirname(__file__))
    #判断inages文件夹是否存在，否则创建images文件夹
    if os.path.exists('images'):
        pass
    else:
        os.mkdir("images")
    #改变图片的保存路径为当前目录的images文件夹下
    file_path = file_path+'/images/'
    #拼接上文件的名称 ，filename为文件的保存名称
    file_path_name = file_path+filename
    return file_path_name

def url_address_dom(num):
    url = r'https://www.vcg.com/ajax/channel/tagitemlist'
    #请求头
    header = {
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    }
    #请求参数
    data = {
        'timeliness': '0',
        'key': '220671',
        'channelId': '0',
        'channelName': '伤心',
        'channelPath': 'Sad',
        'page': num,
        'per_page': '100',
        'isEdit': '1'
    }

    data = parse.urlencode(data).encode('utf-8')
    req = request.Request(url=url,data=data,headers=header)
    open_url = request.urlopen(req).read().decode('utf8')
    url_cha_dic = json.loads(open_url)
    url_adds = url_cha_dic['data']['list']
    for url_add in url_adds:
        url_dom_add ='https:'+url_add['equalh_url']
        filename = url_dom_add.split('/')[-1]
        path_add = path_name(filename)
        image_dom = request.urlopen(url_dom_add).read()
        # image_dom = image_dom.read()
        print('下载中.....',num)
        with open(path_add,'wb') as f:
            f.write(image_dom)

if __name__ == "__main__":
    for i in range(1,5):
        print(i)
        url_address_dom(i)



#
