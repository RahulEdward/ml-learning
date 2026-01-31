# pandas ke saath Efficient data storage

Notebook [storage_benchmark](storage_benchmark.ipynb) (iska Hindi version `storage_benchmark_HINDI.ipynb` bhi hai) efficiency aur performance ke liye main storage formats ko compare karta hai.

Khaas taur par, ye compare karta hai:
- CSV: Comma-separated, standard flat text file format.
- HDF5: Hierarchical data format, jo shuru mein National Center for Supercomputing mein develop kiya gaya tha, numerical data ke liye ek fast aur scalable storage format hai, jo PyTables library ka use karke pandas mein available hai.
- Parquet: Ek binary, columnar storage format jo Apache Hadoop ecosystem ka hissa hai, efficient data compression aur encoding deta hai aur ise Cloudera aur Twitter dwara develop kiya gaya hai. Ye `pyarrow` library ke zariye pandas ke liye available hai, jise pandas ke original author Wes McKinney lead karte hain.

Ye ek test `DataFrame` ka use karta hai jise numerical ya text data, ya dono rakhne ke liye configure kiya ja sakta hai. HDF5 library ke liye, hum fixed aur table dono format test karte hain. Table format queries ki suvidha deta hai aur isme data append kiya ja sakta hai.

### Test Results (Natije)

Sankanxep mein, natije ye hain:
- Sirf numerical data ke liye, HDF5 format sabse accha perform karta hai, aur table format CSV ke saath smallest memory footprint (1.6 GB) share karta hai. Fixed format doguni jagah leta hai, aur parquet format 2 GB use karta hai.
- Numerical aur text data ke mix ke liye, parquet kaafi tezi se kaam karta hai, aur HDF5 CSV ke mukable padhne (reading) mein apna fayda dikhata hai.

Notebook dikhata hai ki `%%timeit` cell magic ka use karke kaise configure, test, aur timing collect karein. Saath hi ye in storage formats ko use karne ke liye zaruri related pandas commands ka upyog bhi dikhata hai.
