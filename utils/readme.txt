Step 1: get the html files corresponding to pages on techexplore with scrape_html.sh .
- e.g. https://techxplore.com/computer-sciences-news/page{i}.html 
- alternatively you can also manually access the content at https://r.jina.ai/https://techxplore.com/computer-sciences-news/page{i}.html, without using any API tokens.


Step 2: extract all urls to news articles with get_html.py

Step 3: Using the urls, scrape the raw news content with Jina (scrape_news.py)
- Some scrapings may fail in the first run. You may have to run a second try by collecting the IDs of failed instances.

Step 4: sanitize the raw news content with clean_news.py