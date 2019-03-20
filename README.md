# CS260Final

### Running Project
- Download Yelp data set from Kaggle. It can be found here: [Yelp on Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset)
- Ingest data into Microsoft SQL Server 2016 or newer
- Run the following two SQL Scripts
    - `SQL\AverageOfFriends.sql`
    - `SQL\Input.sql` saving the output as 'input.csv'

- Run `NN\nn.py` and it will train the model on our best configuration
    - Optionally, one can run run `NN\nn.py <input csv path> <drop last N columns> <fraction of data used>`.