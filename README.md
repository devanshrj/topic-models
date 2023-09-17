# Topic Models

## Steps
1. Use `preprocess.py` to preprocess the data. The script converts a MySQL message table into a CSV file to be read by models. Note that you must manually change the database, table name, and path to store the CSV file in `preprocess.py`. 

2. Use `model.py` to generate topics and user-topic vectors for `{LDA, ProdLDA}`. As before, you must manually change path arguments to direct to the CSV file generated in step 1.
