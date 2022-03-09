import json

import scrapy


class ArticlesSpider(scrapy.Spider):
    name = "articles"

    def start_requests(self):
        urls = []

        with open(self.urls_path) as f:
            data = json.load(f)
            urls.extend(data["urls"])

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f"{page}.html"
        with open(f"{self.articles_dir}/{filename}", "wb") as f:
            f.write(response.body)
