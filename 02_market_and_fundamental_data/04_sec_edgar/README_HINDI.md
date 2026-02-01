
## How to work ke saath Fundamental data

The Securities aur Exchange Commission (SEC) requires US issuers, that hai, listed companies aur securities, including mutual funds to file three quarterly financial statements (Form 10-Q) aur one annual report (Form 10-K), mein addition to various other regulatory filing requirements.

Since the early 1990s, the SEC made these filings available through its Electronic Data Gathering, Analysis, aur Retrieval (EDGAR) system. They constitute the primary data source ke liye the fundamental analysis ka equity aur other securities, such as corporate credit, where the value depends on the business prospects aur financial health ka the issuer. 

#### Automated processing use karke XBRL markup

Automated analysis ka regulatory filings has become much easier since the SEC introduced XBRL, a free, open, aur global standard ke liye the electronic representation aur exchange ka business reports. XBRL hai based on XML; it relies on [taxonomies](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) that define the meaning ka the elements ka a report aur map to tags that highlight the corresponding information mein the electronic version ka the report. One such taxonomy represents the US Generally Accepted Accounting Principles (GAAP).

The SEC introduced voluntary XBRL filings mein 2005 mein response to accounting scandals before requiring this format ke liye all filers since 2009 aur continues to expand the mandatory coverage to other regulatory filings. The SEC maintains a website that lists the current taxonomies that shape the content ka different filings aur can be used to extract specific items.

There hain several avenues to track aur access fundamental data reported to the SEC:
- Ke hisse ke roop mein the [EDGAR Public Dissemination Service]((https://www.sec.gov/oit/announcement/public-dissemination-service-system-contact.html)) (PDS), electronic feeds ka accepted filings hain available ke liye a fee. 
- The SEC updates [RSS feeds](https://www.sec.gov/structureddata/rss-feeds-submitted-filings) every 10 minutes, which list structured disclosure submissions.
- There hain public [index files](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm) ke liye the retrieval ka all filings through FTP ke liye automated processing.
- The financial statement (aur notes) datasets contain parsed XBRL data from all financial statements aur the accompanying notes.

The SEC also publishes log files containing the [internet search traffic](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) ke liye EDGAR filings through SEC.gov, albeit ke saath a six-month delay.


#### Building a fundamental data time series

The scope ka the data mein the [Financial Statement aur Notes](https://www.sec.gov/dera/data/financial-statement-aur-notes-data-set.html) datasets consists ka numeric data extracted from the primary financial statements (Balance sheet, income statement, cash flows, changes mein equity, aur comprehensive income) aur footnotes on those statements. The data hai available as early as 2009.


The folder [03_sec_edgar](03_sec_edgar) contain karta hai the notebook [edgar_xbrl](03_sec_edgar/edgar_xbrl.ipynb) to download aur parse EDGAR data mein XBRL format, aur create fundamental metrics like the P/E ratio by combining financial statement aur price data.

### Other fundamental data sources

- [Compilation ka macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-aur-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)
