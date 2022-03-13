from dataclasses import dataclass

from bs4 import BeautifulSoup, element
from unidecode import unidecode


@dataclass
class SourceLink:
    link: str
    link_text: str


@dataclass
class Source:
    text: str
    links: list[SourceLink]


class BSParser:
    def __init__(self, filepath: str, url: str):
        self.soup = BeautifulSoup(open(filepath), "html.parser")
        self.filepath = filepath
        self.url = url
        self.text_tags = ["b", "strong", "font", "div", "u", "em", "i", "span"]
        self.non_source_tags = ["img", "script", "table", "center"]

    def __parse_header(self):
        statements = self.soup.find_all("article", class_="m-statement--is-xlarge")
        assert len(statements) == 1
        statement = statements.pop()

        author = statement.find("a", class_="m-statement__name").getText()
        if not author:
            print(f"empty author for {self.url}")

        claim_desc = statement.find("div", class_="m-statement__desc").getText()
        if not claim_desc:
            print(f"empty claim_desc for {self.url}")

        claim = statement.find("div", class_="m-statement__quote").getText()
        if not claim:
            print(f"empty claim for {self.url}")

        label = (
            statement.find("div", class_="m-statement__meter")
            .find("picture")
            .find("img")
            .get("alt")
        )
        if not label:
            print(f"empty label for {self.url}")

        return (
            unidecode(author.strip()),
            unidecode(claim_desc.strip()),
            unidecode(claim.strip()),
            unidecode(label.strip()),
        )

    def __parse_p_separated_sources(self, paragraph):
        links = paragraph.find_all("a")
        resource_text = unidecode(paragraph.getText()).strip()

        if not resource_text:
            return []

        sources = []
        if len(links) == 0:
            if ", 20" in resource_text and len(resource_text) < 14:
                # <p> containing only date // did-ann-wagner-vote-repeal-pre-existing-conditions.html
                print(f'will not save date: "{resource_text}" from {self.url}')
            elif resource_text:
                sources.append(vars(Source(text=resource_text, links=[])))
        else:
            source_links = []

            for link in links:
                if (
                    ", 20" not in link.getText() or len(link.getText()) > 14
                ):  # don't want <a> July 29, 2008 </a>
                    source_links.append(
                        vars(
                            SourceLink(
                                link=link.get("href"),
                                link_text=unidecode(link.getText()),
                            )
                        )
                    )
                else:
                    print(f'will not save link: "{link.getText()}" from {self.url}')

            sources.append(vars(Source(text=resource_text, links=source_links)))

        return sources

    def __create_sources(self, content, links, link_texts):
        source_links = []
        sources = []

        resource_text = unidecode("".join(content).strip())

        if resource_text:
            if len(links) > 0:
                for i in range(len(links)):
                    source_links.append(
                        vars(
                            SourceLink(
                                link=links[i], link_text=unidecode(link_texts[i])
                            )
                        )
                    )
            else:
                source_links.append(vars(SourceLink(link="", link_text="")))

            sources.append(vars(Source(text=resource_text, links=source_links)))

        return sources

    def __parse_a_tag(self, tag):
        assert tag.name == "a"

        link = tag.get("href") if tag.get("href") else ""
        link_text = unidecode(tag.getText())

        return link, link_text

    def __parse_text_tags(self, tag):
        sources = []

        content = []
        links = []
        link_texts = []

        if len(tag.find_all("a")) == 0 and len(tag.find_all("br")) == 0:
            return tag.getText()
        else:
            for b_child in tag.children:
                if b_child.name == "br":
                    new_sources = self.__create_sources(
                        content=content, links=links, link_texts=link_texts
                    )
                    sources.extend(new_sources)

                    content = []
                    links = []
                    link_texts = []
                elif b_child.name == "a":
                    link, link_text = self.__parse_a_tag(b_child)

                    links.append(link)
                    link_texts.append(link_text)
                    content.append(link_text)
                elif isinstance(b_child, str):
                    content.append(b_child)
                elif b_child.name in self.text_tags:
                    parsed = self.__parse_text_tags(b_child)

                    if isinstance(parsed, str):
                        content.append(parsed)
                    else:
                        sources.extend(parsed)
                else:
                    print(
                        "some strange case inside <b>, <strong> or <font> tag",
                        self.url,
                    )
                    print(b_child)

        if len(content) > 0:
            new_sources = self.__create_sources(
                content=content, links=links, link_texts=link_texts
            )
            sources.extend(new_sources)

        return sources

    def __parse_br_separated_sources(self, paragraph, has_special_div=False):
        sources = []

        content = []
        links = []
        link_texts = []

        for a in paragraph.children:
            if a.name == "br":
                new_sources = self.__create_sources(
                    content=content, links=links, link_texts=link_texts
                )
                sources.extend(new_sources)

                content = []
                links = []
                link_texts = []
            elif a.name == "a":
                link, link_text = self.__parse_a_tag(a)

                links.append(link)
                link_texts.append(link_text)
                content.append(link_text)
            elif (
                a.name != "a"
                and isinstance(a, element.Tag)
                and len(a.find_all("a")) == 1
                and len(a.find_all("br")) == 0
            ):
                a = a.find("a")

                link, link_text = self.__parse_a_tag(a)

                links.append(link)
                link_texts.append(link_text)
                content.append(link_text)
            elif isinstance(a, str):
                content.append(a)
            elif a.name == "div" and has_special_div:
                new_sources = self.__parse_div_tag(a)
                sources.extend(new_sources)
            elif a.name in self.text_tags:
                parsed = self.__parse_text_tags(a)

                if isinstance(parsed, str):
                    content.append(parsed)
                else:
                    sources.extend(parsed)
            elif a.name not in self.non_source_tags:
                print("some strange case", self.url)
                print(a)  # no such cases for now

        new_sources = self.__create_sources(
            content=content, links=links, link_texts=link_texts
        )
        sources.extend(new_sources)

        return sources

    def __parse_div_tag(self, tag):
        sources = []

        links = []
        link_texts = []
        content = []

        if len(tag.find_all("p")) == 0:
            new_sources = self.__parse_br_separated_sources(tag, has_special_div=True)
            sources.extend(new_sources)
        else:
            for child in tag.children:
                if child.name == "p":
                    new_sources = []
                    if len(child.find_all("br")) > 0:
                        new_sources = self.__parse_br_separated_sources(child)
                    else:
                        new_sources = self.__parse_p_separated_sources(child)
                        print("br inside p inside div", self.url)
                    sources.extend(new_sources)
                elif child.name == "div":
                    new_sources = self.__parse_div_tag(child)
                    sources.extend(new_sources)
                elif not child.getText().strip():
                    continue
                elif isinstance(child, str):
                    content.append(child)
                elif child.name == "a":
                    link, link_text = self.__parse_a_tag(child)

                    links.append(link)
                    link_texts.append(link_text)
                    content.append(link_text)
                else:
                    print("strange child inside div", self.url)

        if len(content) > 0:
            new_sources = self.__create_sources(
                content=content, links=links, link_texts=link_texts
            )
            sources.extend(new_sources)

        return sources

    def __parse_sources(self):
        resources_sections = self.soup.find_all("section", id="sources")
        assert len(resources_sections) == 1

        sources_section = resources_sections.pop().find("article")

        sources = []

        content = []
        link = ""  # check for multiple links here
        link_text = ""

        for child in sources_section.children:
            if child.name == "p":
                # create sources from the current accumulated content
                new_sources = self.__create_sources(
                    content=content, links=[link], link_texts=[link_text]
                )
                sources.extend(new_sources)

                content = []
                link = ""
                link_text = ""

                # parse the current tag
                parsed = []

                if (
                    len(child.find_all("br")) > 0
                ):  # all sources are on one <p> separated by <br>
                    parsed = self.__parse_br_separated_sources(child)
                else:
                    parsed = self.__parse_p_separated_sources(
                        child
                    )  # one source per <p>

                sources.extend(parsed)
            elif child.name == "a":
                if len(content) > 0:
                    new_sources = self.__create_sources(
                        content=content, links=[link], link_texts=[link_text]
                    )
                    sources.extend(new_sources)

                    content = []
                    link = ""
                    link_text = ""

                if child.get("href"):
                    if link:
                        print("will replcase link", self.url)
                    link = child.get("href")

                if link_text:
                    print("will replcase link_text", self.url)
                link_text = child.getText()

                content.append(link_text)
            elif child.name == "div":
                new_sources = self.__parse_div_tag(child)
                sources.extend(new_sources)
            elif (
                child.name != "a"
                and isinstance(child, element.Tag)
                and len(child.find_all("a")) == 1  # check if we have multiple <a>
            ):
                child = child.find("a")

                if link:
                    print("will replcase link", self.url)

                if link_text:
                    print("will replcase link_text", self.url)

                link = child.get("href")
                link_text = child.getText()
                content.append(link_text)
            elif child.name == "ul":
                if len(child.find_all("a")) == 0:
                    content.append(child.getText())
                else:
                    for list_element in child.find_all("li"):
                        p_tags = list_element.find_all("p")
                        parsed = []
                        if len(p_tags) == 0:
                            parsed = self.__parse_p_separated_sources(list_element)
                        elif len(p_tags) == 1:
                            parsed = self.__parse_p_separated_sources(p_tags.pop())
                        else:
                            print("multiple <p> inside <li>", self.url)
                        sources.extend(parsed)
            elif (
                child.name != "a"
                and isinstance(child, element.Tag)
                and len(child.find_all("a")) > 1  # check if we have multiple <a>
            ):
                print("multiple <a> inside sources_section.children", self.url)
            elif (
                child.name in self.text_tags
            ):  # make sure these do not contain <a> or <br>
                if len(child.find_all("a")) == 0 and len(child.find_all("br")) == 0:
                    content.append(child.getText())
                else:
                    print("child in self.text_tags with <a/br>", self.url)
            elif not child.getText().strip():
                continue
            elif isinstance(child, str):
                content.append(child)
            elif child.name not in self.non_source_tags:
                print("some strange case", self.url)
                print(child)
                print("-------------------------------------------------------------")

        if len(content) > 0:
            new_sources = self.__create_sources(
                content=content, links=[link], link_texts=[link_text]
            )
            sources.extend(new_sources)

        return sources

    def parse(self):
        sources = self.__parse_sources()
        author, claim_desc, claim, label = self.__parse_header()

        return {
            "url": self.url,
            "author": author,
            "claim_desc": claim_desc,
            "claim": claim,
            "label": label,
            "sources": sources,
        }
