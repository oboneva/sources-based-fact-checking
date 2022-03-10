import scrapy
from bs4 import BeautifulSoup


class UrlsSpider(scrapy.Spider):
    name = "urls"

    def start_requests(self):
        for i in range(1, 64):
            url = f"https://www.politifact.com/factchecks/list/?page={i}"
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        soup = BeautifulSoup(response.body, "html.parser")

        items_lists = soup.find_all("ul", class_="o-listicle__list")
        assert len(items_lists) == 1

        items_list = items_lists.pop()

        urls = []
        for item in items_list.find_all("li", class_="o-listicle__item"):
            statements = item.find_all("div", class_="m-statement__quote")
            assert len(statements) == 1

            statement = statements.pop()
            links = statement.find_all("a")
            assert len(links) == 1

            link = links.pop()
            href = link.get("href")

            if href:
                url = f"https://www.politifact.com{href}"
                urls.append(url)
            else:
                print("missing link on page", response.url)

        yield {"page": urls}
