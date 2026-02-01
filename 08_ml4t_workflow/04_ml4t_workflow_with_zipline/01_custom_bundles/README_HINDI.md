# Zipline: Ingesting custom minute data

The `python` scripts mein this directory sketch how to ingest custom minute data mein Zipline. It hai based on the Algoseek minute-bar trade data, which hai not available ke liye free. 
However, you can create a similar dataset by extracting the first, high, low, last aur volume columns from the free sample ka trade-aur-quote data generously provided by [Algoseek](https://www.algoseek.com) [here](https://www.algoseek.com/ml4t-book-data.html).

Unfortunately, Zipline's pipeline API does not work ke liye minute-bar data, so hum hain not use karke this custom bundle mein the book but I am leaving this sample code here ke liye adapation to your own prjects.
