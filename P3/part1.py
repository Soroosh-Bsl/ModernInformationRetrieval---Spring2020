from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from MIRP3.spiders.scholar_spider import ScholarSpider


def scrap(desired_number_of_papers=2000, start_urls=None):
    if start_urls is None:
        start_urls = ["https://www.semanticscholar.org/paper/The-Lottery-Ticket-Hypothesis%3A-Training-Pruned-Frankle-Carbin/f90720ed12e045ac84beb94c27271d6fb8ad48cf",
                  "https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                  "https://www.semanticscholar.org/paper/BERT%3A-Pre-training-of-Deep-Bidirectional-for-Devlin-Chang/df2b0e26d0599ce3e70df8a9da02e51594e0e992"]

    # spider = ScholarSpider(desired_number_of_papers=desired_number_of_papers, start_urls=start_urls)
    process = CrawlerProcess(get_project_settings())
    process.crawl(ScholarSpider, start_urls=start_urls, desired_number_of_papers=desired_number_of_papers)
    process.start()


