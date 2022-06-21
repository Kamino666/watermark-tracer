# encoding:utf-8
import requests
import base64
import json
import io
from PIL import Image
import json
import sys

from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.signalmanager import dispatcher

sys.path.append('searchengine')
from searchengine.spiders.bing import BingSpider
from searchengine.spiders.baidu import BaiduSpider
from searchengine.spiders.ss_360 import Ss360Spider


class BaiduAPI:
    def __init__(self, config: str):
        with open(config) as f:
            data = json.load(f)
        host = f"https://aip.baidubce.com/oauth/2.0/token?" \
               f"grant_type=client_credentials&client_id={data['ak']}&client_secret={data['sk']}"
        response = requests.get(host)
        self.access_token = response.json()['access_token']

    def detect_mark(self, image: Image.Image):
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/logo"
        params = {"custom_lib": "false", "image": self.image_to_base64(image)}
        request_url = request_url + "?access_token=" + self.access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        return response.json()

    def detect_text(self, image: Image.Image):
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/webimage"
        params = {"image": self.image_to_base64(image)}
        request_url = request_url + "?access_token=" + self.access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        return response.json()

    def search(self, text: str):
        result = self.spider_results('bing', text, 1, 1)
        if not result:
            result = self.spider_results('ss_360', text, 1, 1)
        if not result:
            result = self.spider_results('baidu', text, 1, 1)
        if not result:
            return None
        return json.loads(result)

    @staticmethod
    def image_to_base64(image: Image.Image, fmt='png') -> str:
        output_buffer = io.BytesIO()
        image.save(output_buffer, format=fmt)
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode('utf-8')
        return base64_str

    def spider_results(self, spidername, keywords, pagenum, sorttype):
        spider_class = None
        if spidername == 'bing':
            spider_class = BingSpider
        elif spidername == 'baidu':
            spider_class = BaiduSpider
        elif spidername == "ss_360":
            spider_class = Ss360Spider
        else:
            return []

        results = []

        def crawler_results(signal, sender, item, response, spider):
            results.append(dict(item))

        dispatcher.connect(crawler_results, signal=signals.item_passed)

        process = CrawlerProcess(get_project_settings())
        process.crawl(spider_class, keywords=keywords,
                      pagenum=pagenum, sorttype=sorttype)
        process.start()  # the script will block here until the crawling is finished
        return json.dumps(results, ensure_ascii=False).encode('gbk', 'ignore').decode('gbk')


if __name__ == '__main__':
    api = BaiduAPI("baidu_cfg_sample.json")
    x = api.search('烤鱼official')
