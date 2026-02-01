# Custom Bundle ke liye Japanese Stocks sourced from STOOQ

hum hain going to create a [custom bundle](https://zipline.ml4trading.io/bundles.html#writing-a-new-bundle) ke liye `Zipline` use karke Japanese equity data; see [download instructions](../../data/create_stooq_data.ipynb) first.  

hum will take the following steps:
1. Create several data files containing information on tickers, prices, aur adjustments
2. Code up a `Zipline` ingest function that handles the data processing aur storage
3. Define a `Zipline` extension that registers the new `bundle`
4. Place the files mein the `Zipline_ROOT` directory to ensure the `Zipline ingest` command finds them

## Setup

`Zipline` permits the creation ka custom bundle containing open, high, low, close aur volume (OHCLV) information, as well as adjustments like stock splits aur dividend payments.

It stores the data per default a `.Zipline` directory mein the user's home directory, `~/.Zipline`. However, you can modify the target location by setting the `Zipline_ROOT` environment variable as hum do ke liye the docker images provided ke saath this book.   

## Data preprocessing

To prepare the data, hum create three kinds ka data tables mein HDF5 format:
1. `equities`: contain karta hai a unique `sid`, the `ticker`, aur a `name` ke liye the security.
2. price tables ke saath OHLCV data ke liye each ka the ~2,900 assets, named `jp.<sid>`
3. `splits`: contain karta hai split factors aur hai required; our data hai already adjusted so hum just add one line ke saath a factor ka 1.0 ke liye one   

The file `stooq_preprocessing` implements these steps aur produces the tables mein the HDF5 file `stooq.h5`.

## `Zipline` ingest function

The file `stooq_jp_stocks.py` defines a function `stooq_jp_to_bundle(interval='1d')` that returns the `ingest` function required by `Zipline` to produce a custom bundle (see [docs](https://zipline.ml4trading.io/bundles.html#writing-a-new-bundle). It needs to have the following signature:

```python
ingest(environ,
       asset_db_writer,
       minute_bar_writer,
       daily_bar_writer,
       adjustment_writer,
       calendar,
       start_session,
       end_session,
       cache,
       show_progress,
       output_dir)
```

Yeh function loads the information hum crated mein the previous step during the `ingest` process. It consists ka a `data_generator()` that loads `(sid, ticker)` tuples as needed, aur produces the corresponding OHLCV info mein the correct format. It also adds information about the exchange so Zipline can associate the right calendar, aur the range ka trading dates.

It also loads the adjustment data, which mein this case does not play an active role.

## Bundle registration

Zipline needs to know that the bundle exists aur how to create the `ingest` function hum just defined. To this end, hum create an `extension.py` file that communicates the bundle's name, where to find the function that returns the `ingest` function (namely `stooq_jp_to_bundle()` mein `stooq_jp_stocks.py`), aur indicates the trading calendar to use (`XTKS` ke liye Tokyo's exchange).

## File locations

Finally, hum need to put these files mein the right locations so that Zipline finds them. hum can use symbolic links while keeping the actual files mein this directory.

More specifically, hum'll create symbolic links to 
1. to `stooq_jp_stocks.py` mein the ZIPLINE_ROOT directory, aur 
2. to stooq.h5 mein `ZIPLINE_ROOT/custom_data`

mein Linux or MacOSX, this implies opening the shell aur running the following commands (where PROJECT_DIR refers to absolute path to the root folder ka this repository on your machine)
```bash
cd $ZIPLINE_ROOT
ln -s PROJECT_DIR/11_decision_trees_random_forests/00_custom_bundle/stooq_jp_stocks.py
ln -s PROJECT_DIR/machine-learning-ke liye-trading/11_decision_trees_random_forests/00_custom_bundle/extension.py .
mkdir custom_data
ln -s PROJECT_DIR/11_decision_trees_random_forests/00_custom_bundle/stooq.h5 custom_data/.
``` 

As a result, your directory structure should look as follows (some ka these files will be symbolic links):
```python
ZIPLINE_ROOT
    |-extension.py
    |-stooq_jp_stocks.py
    |-custom_data
        |-stooq.h5
```


