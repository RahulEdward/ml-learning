import json
from pathlib import Path

# Path to the original notebook
nb_path = Path(r'd:/machine-learning-for-trading-main/02_market_and_fundamental_data/01_NASDAQ_TotalView-ITCH_Order_Book/01_parse_itch_order_flow_messages.ipynb')
output_path = Path(r'd:/machine-learning-for-trading-main/02_market_and_fundamental_data/01_NASDAQ_TotalView-ITCH_Order_Book/01_parse_itch_order_flow_messages_HINDI.ipynb')

# Translations mapped by cell index
translations = {
    0: [
        "# Order Book Data ke saath kaam karna: NASDAQ ITCH"
    ],
    1: [
        "Market data ka main source 'order book' hai, jo din bhar real-time mein lagatar update hoti rehti hai taaki trading ki saari activity dikhayi de sake. Exchanges aamtaur par ye data real-time service ke roop mein dete hain aur kuch purana (historical) data free mein bhi de sakte hain.\n",
        "\n",
        "Trading activity un dher saare messages mein dikhti hai jo market participants trade orders ke liye bhejte hain. Ye messages aamtaur par electronic Financial Information eXchange (FIX) communications protocol (jo securities transactions aur market data ke real-time exchange ke liye hota hai) ya phir exchange ke apne native protocol ke hisab se hote hain."
    ],
    2: [
        "## Background (Poshthbhoomi)"
    ],
    3: [
        "### The FIX Protocol"
    ],
    4: [
        "Jaise SWIFT back-office messaging (jaise trade-settlement) ke liye ek message protocol hai, waise hi [FIX protocol](https://www.fixtrading.org/standards/) exchanges, banks, brokers, clearing firms aur dusre market participants ke beech trade execution se pehle aur uske dauran batchit karne ka standard hai. Fidelity Investments aur Salomon Brothers ne 1992 mein FIX shuru kiya tha taaki broker-dealers aur institutional clients ke beech electronic communication ho sake, jo us waqt tak phone par baat karte the.\n",
        "\n",
        "Yeh pehle global equity markets mein popular hua aur phir foreign exchange, fixed income aur derivatives markets tak fail gaya, aur baad mein post-trade processing mein bhi use hone laga. Exchanges FIX messages ko ek real-time data feed ke roop mein dete hain jise algorithmic traders analyze karte hain taaki wo market activity ko track kar sakein aur, example ke liye, market participants ke footprint ko pehchan sakein aur unke agle kadam ka andaza laga sakein."
    ],
    5: [
        "### Nasdaq TotalView-ITCH Order Book data"
    ],
    6: [
        "Halaanki FIX ka market share bahut bada hai, exchanges apne khud ke native protocols bhi offer karte hain. Nasdaq ek [TotalView ITCH direct data-feed protocol](http://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf) deta hai jo subscribers ko allow karta hai ki wo:\n",
        "equity instruments ke individual orders ko track kar sakein - order lagane se lekar uske execute hone ya cancel hone tak.\n",
        "\n",
        "Natijan, ye order book ko phir se banane (reconstruction) mein madad karta hai jo kisi specific security ya financial instrument ke active buy aur sell orders ki list ka hisab rakhta hai. Order book har price point par kitne shares ki boli (bid) ya offer lagayi gayi hai, ye dikhakar din bhar ki market depth batata hai. Ye specific buy aur sell orders ke piche kaunsa market participant hai, ye bhi pehchan sakta hai (agar order anonymously nahi lagaya gaya ho). Market depth liquidity ka aur bade market orders ka price par kya asar padega, iska ek main indicator hai."
    ],
    7: [
        "ITCH v5.0 specification mein 20 se zyada message types hain jo system events, stock characteristics, limit orders lagane aur badalne, aur trade execution se jude hain. Isme open aur closing cross se pehle net order imbalance ki jaankari bhi hoti hai."
    ],
    8: [
        "## Imports"
    ],
    13: [
        "## FTP Server se NASDAQ ITCH Data lena"
    ],
    14: [
        "Nasdaq kai mahino ke daily binary files ke [samples](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/) offer karta hai.\n",
        "\n",
        "Ab hum dikhayenge ki ITCH messages ki sample file ko kaise parse karein aur kisi bhi diye gaye tick ke liye executed trades aur order book dono ko kaise reconstruct karein."
    ],
    15: [
        "Data kaafi bada hai aur poore example ko chalane mein bahut samay lag sakta hai aur kaafi memory (16GB+) ki zarurat pad sakti hai. Saath hi, is example mein use ki gayi sample file shayad ab available na ho kyunki NASDAQ kabhi-kabhi sample files update karta rehta hai."
    ],
    16: [
        "Niche di gayi table March 29, 2018 ki sample file ke liye sabse aam message types ki frequency dikhati hai:"
    ],
    17: [
        "| Name                    | Offset  | Length  | Value      | Notes                                                                                |\n",
        "|-------------------------|---------|---------|------------|--------------------------------------------------------------------------------------|\n",
        "| Message Type            | 0       | 1       | S          | System Event Message                                                                 |\n",
        "| Stock Locate            | 1       | 2       | Integer    | Always 0                                                                             |\n",
        "| Tracking Number         | 3       | 2       | Integer    | Nasdaq internal tracking number                                                      |\n",
        "| Timestamp               | 5       | 6       | Integer    | Nanoseconds since midnight                                                           |\n",
        "| Order Reference Number  | 11      | 8       | Integer    | Naye order ko receipt ke waqt diya gaya unique reference number.                     |\n",
        "| Buy/Sell Indicator      | 19      | 1       | Alpha      | Order ka type. B = Buy Order. S = Sell Order.                                        |\n",
        "| Shares                  | 20      | 4       | Integer    | Book mein add kiye jane wale order ke total shares.                                  |\n",
        "| Stock                   | 24      | 8       | Alpha      | Stock symbol, right padded with spaces                                               |\n",
        "| Price                   | 32      | 4       | Price (4)  | New order ka display price. Field processing notes ke liye Data Types dekhein.       |\n",
        "| Attribution             | 36      | 4       | Alpha      | Entered order se juda Nasdaq Market participant identifier                           |"
    ],
    18: [
        "### Data paths set karein"
    ],
    19: [
        "Hum download ko `data` subdirectory mein store karenge aur result ko `hdf` format mein convert karenge (jiske baare mein chapter 2 ke aakhri section mein bataya gaya hai)."
    ],
    21: [
        "Aap [NASDAQ server](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/) par kai sample files dhundh sakte hain.\n",
        "\n",
        "Is example mein use kiya gaya HTTPS address, filename aur corresponding date:"
    ],
    23: [
        "#### URL updates\n",
        "\n",
        "NASDAQ kabhi-kabhi files update karta hai jisse SOURCE_FILE badal jata hai. Agar upar wala code error deta hai, to apne browser mein HTTPS_URL par jayen aur nayi files check karein. September 2021 tak, listed files mein ye shamil hain:\n",
        "\n",
        "- 01302020.NASDAQ_ITCH50.gz\n",
        "- 12302019.NASDAQ_ITCH50.gz\n",
        "- 10302019.NASDAQ_ITCH50.gz\n",
        "- 08302019.NASDAQ_ITCH50.gz\n",
        "- 07302019.NASDAQ_ITCH50.gz\n",
        "- 03272019.NASDAQ_ITCH50.gz\n",
        "- 01302019.NASDAQ_ITCH50.gz\n",
        "- 12282018.NASDAQ_ITCH50.gz\n"
    ],
    24: [
        "### Download & unzip"
    ],
    26: [
        "Yeh 5.1GB data download karega jo unzip hone par 12.9GB ho jata hai (yeh file ke hisab se alag ho sakta hai, niche 'url updates' dekhein)."
    ],
    28: [
        "## ITCH Format Settings"
    ],
    29: [
        "### Binary data ke liye `struct` module"
    ],
    30: [
        "ITCH tick data binary format mein aata hai. Python `struct` module (dekhein [docs](https://docs.python.org/3/library/struct.html)) provide karta hai binary data ko parse karne ke liye. Yeh format strings ka use karta hai jo specification mein bataye gaye tarike se byte string ke alag-alag components ki length aur type batakar message elements ki pehchan karte hain."
    ],
    31: [
        "Docs se:\n",
        "\n",
        "> Ye module Python values aur C structs (jo Python bytes objects ke roop mein hote hain) ke beech conversion karta hai. Iska use files mein store kiye gaye ya network connections se aane wale binary data ko handle karne ke liye kiya ja sakta hai. Yeh Format Strings ko use karta hai C structs ke layout aur Python values mein conversion ko describe karne ke liye."
    ],
    32: [
        "Chaliye trading messages ko parse karne aur order book ko reconstruct karne ke critical steps ko dekhte hain:"
    ],
    33: [
        "### Format strings define karma"
    ],
    34: [
        "Parser niche diye gaye formats dictionaries ke hisab se format strings ka use karta hai:"
    ],
    38: [
        "### Binary data parser ke liye message specs create karna"
    ],
    39: [
        "ITCH parser message specifications par nirbhar karta hai jo hum agle steps mein banayenge."
    ],
    40: [
        "#### Message Types Load karein"
    ],
    41: [
        "`message_types.xlxs` file mein message type specs hain jaisa ki [documentation](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf) mein diya gaya hai."
    ],
    44: [
        "#### Basic Cleaning"
    ],
    45: [
        "`clean_message_types()` function bas kuch basic string cleaning steps chalata hai."
    ],
    48: [
        "#### Message Labels Prapt karein"
    ],
    49: [
        "Hum message type codes aur names extract karte hain taaki hum baad mein results ko padhne mein aasan bana sakein."
    ],
    51: [
        "### Specification details ko finalize karein"
    ],
    52: [
        "Har message mein kai fields hote hain jo offset, length aur value ke type se define hote hain. `struct` module binary source data ko parse karne ke liye is format information ka use karega."
    ],
    55: [
        "Optionally, file se persist/reload karein:"
    ],
    58: [
        "Parser message specs ko format strings aur `namedtuples` mein translate karta hai jo message content ko capture karte hain. Sabse pehle, hum ITCH specs se `(type, length)` formatting tuples banate hain:"
    ],
    60: [
        "Iske baad, hum alphanumerical fields ke liye formatting details extract karte hain"
    ],
    62: [
        "Hum message classes ko named tuples aur format strings ke roop mein generate karte hain"
    ],
    66: [
        "`alpha` type (alphanumeric) waale fields ko post-processing ki zarurat hoti hai jaisa ki `format_alpha` function mein defined hai:"
    ],
    68: [
        "## Binary Message Data ko Process karein"
    ],
    69: [
        "Ek din ki binary file mein 350,000,000 se zyada messages hote hain jo 12 GB se zyada ke hote hain."
    ],
    72: [
        "Script parsed result ko iteratively fast HDF5 format waali file mein append karta hai `store_messages()` function ka use karke (jise humne abhi define kiya tha) taaki memory constraints se bacha ja sake (is format ke baare mein chapter 2 ke aakhri section mein aur dekhein)."
    ],
    73: [
        "Niche diya gaya code binary file ko process karta hai aur message type ke hisab se store kiye gaye parsed orders produce karta hai:"
    ],
    75: [
        "## Trading Day ko Summarize karein"
    ],
    76: [
        "### Trading Message Frequency"
    ],
    79: [
        "### Traded Value ke hisab se Top Equities"
    ]
}

# Read the notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Inject translations
count = 0
for i, cell in enumerate(nb['cells']):
    if i in translations:
        cell['source'] = translations[i]
        count += 1

# Write to new notebook
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully processed {count} translations and saved to {output_path}")
