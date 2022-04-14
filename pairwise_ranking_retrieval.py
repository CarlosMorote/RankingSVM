from packaging import version
from platform import python_version
from dataset import RankingDataset
from SVM import RankSVM

if __name__=='__main__':
    # Args
    min_python_version = '3.8.0'
    if version.parse(python_version()) < version.parse(min_python_version):
        raise Exception(f"The min version of Python must be {min_python_version}")

    data = RankingDataset(datapath='./data/docs.csv', datapath_queries='./data/queries.csv', datapath_query_doc='./data/query_doc.csv')
    rankingModel = RankSVM(data)

    while (query := input("Query (type q to quit): ").strip().lower()) != 'q':
        rankingModel.show_ranking(query, top=10)