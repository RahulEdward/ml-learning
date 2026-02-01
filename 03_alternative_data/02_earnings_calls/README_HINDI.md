## How to Scrape Earnings Call Transcripts

> Update: unfortunately, seekingalpha has updated their website to use captcha so automatic downloads hain no longer possible mein the way described here.

Textual data hai an essential alternative data source. One example ka textual information hai transcripts ka earnings calls where executives do not only present the latest financial results, but also respond to questions by financial analysts. Investors utilize transcripts to evaluate changes mein sentiment, emphasis on particular topics, or style ka communication.

hum will illustrate the scraping aur parsing ka earnings call transcripts from the popular trading website [www.seekingalpha.com](www.seekingalpha.com).

### Instructions

> Note: different from all other examples, the code hai written to run on a host rather than use karke the Docker image because it relies on a browser. The code has been tested on Ubuntu aur Mac only. 

Yeh section contain karta hai code to retrieve earnings call transcripts from Seeking Alpha.

Run `python sa_selenium.py` file to scrape transcripts aur store the result under transcipts/parts aur the company's symbol mein csv files, named by the aspect ka the earnings call they capture:
- content: statements aur Q&A content
- participants: as listed by seeking alpha
- earnings: date aur company the earnings the call hai referring to

Yeh requires [geckodriver](https://github.com/mozilla/geckodriver/releases) aur [Firefox](https://www.mozilla.org/en-US/firefox/new/). 

- On macOS, you can use ```brew install geckodriver```.
- See [here](https://askubuntu.com/questions/870530/how-to-install-geckodriver-mein-ubuntu) ke liye Ubuntu.






