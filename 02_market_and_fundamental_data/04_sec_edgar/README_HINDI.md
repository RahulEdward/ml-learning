# Fundamental data ke saath kaam kaise karein

Securities and Exchange Commission (SEC) sabhi US issuers, yani listed companies aur securities (mutual funds sahit), ke liye zaruri karta hai ki wo teen quarterly financial statements (Form 10-Q) aur ek annual report (Form 10-K) file karein. Iske alawa aur bhi kai regulatory filing requirements hoti hain.

1990s ki shuruwat se, SEC ne apne Electronic Data Gathering, Analysis, and Retrieval (EDGAR) system ke zariye in filings ko available karaya hai. Ye equity aur anya securities (jaise corporate credit) ke fundamental analysis ke liye primary data source hain, jahan value business prospects aur issuer ki financial health par nirbhar karti hai.

#### XBRL markup ka use karke Automated processing

Jab se SEC ne XBRL introduce kiya hai, regulatory filings ka automated analysis bahut aasan ho gaya hai. XBRL business reports ke electronic representation aur exchange ke liye ek free, open, aur global standard hai. XBRL XML par based hai; ye [taxonomies](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) par nirbhar karta hai jo report ke elements ka matlab define karti hain aur unhe tags se map karti hain jo electronic version mein corresponding information ko highlight karte hain. Aisi hi ek taxonomy US Generally Accepted Accounting Principles (GAAP) ko represent karti hai.

SEC ne 2005 mein accounting scandals ke jawab mein voluntary XBRL filings shuru ki thi, aur 2009 se sabhi filers ke liye is format ko zaruri kar diya. Ab wo dusri regulatory filings ke liye bhi mandatory coverage bada raha hai. SEC ek website maintain karta hai jo current taxonomies ki list deti hai, jo alag-alag filings ke content ko shape karti hain aur specific items extract karne ke liye use ki ja sakti hain.

SEC ko report kiye gaye fundamental data ko track aur access karne ke kai tarike hain:
- [EDGAR Public Dissemination Service]((https://www.sec.gov/oit/announcement/public-dissemination-service-system-contact.html)) (PDS) ke hisse ke roop mein, accepted filings ki electronic feeds fee dekar available hain.
- SEC har 10 minute mein [RSS feeds](https://www.sec.gov/structureddata/rss-feeds-submitted-filings) update karta hai, jo structured disclosure submissions list karti hain.
- Automated processing ke liye FTP ke zariye sabhi filings retrieve karne ke liye public [index files](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm) hain.
- Financial statement (aur notes) datasets mein sabhi financial statements aur accompanying notes se parsed XBRL data hota hai.

SEC log files bhi publish karta hai jisme EDGAR filings ke liye [internet search traffic](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) hota hai (SEC.gov ke zariye), halaanki isme cheh mahine ki deri (delay) hoti hai.

#### Fundamental data ki time series banana

[Financial Statement and Notes](https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html) datasets mein data ka scope numeric data hota hai jo primary financial statements (Balance sheet, income statement, cash flows, changes in equity, aur comprehensive income) aur un statements ke footnotes se extract kiya jata hai. Data 2009 se available hai.

Folder [03_sec_edgar](03_sec_edgar) mein notebook [edgar_xbrl](03_sec_edgar/edgar_xbrl.ipynb) hai (iska Hindi version `edgar_xbrl_HINDI.ipynb` bhi hai) jo EDGAR data ko XBRL format mein download aur parse karne ke liye hai, aur financial statement aur price data ko combine karke P/E ratio jaise fundamental metrics create karta hai.

### Anya fundamental data sources

- [Compilation of macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-and-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)
