import argparse
from packaging import version
from platform import python_version
from dataset import RankingDataset
from SVM import RankSVM

if __name__=='__main__':
    min_python_version = '3.8.0'
    if version.parse(python_version()) < version.parse(min_python_version):
        raise Exception(f"The min version of Python must be {min_python_version}")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("datapath_documents",
                help="Filepath to the file containing the collection of documents.")
    parser.add_argument("datapath_queries", help="Filepath to the file containing the collection of queries.")
    parser.add_argument("datapath_query_doc", help="Filepath to the file containing the per-query-per-document ranks.")
    parser.add_argument("--sentence_transformer", default="pritamdeka/S-Biomed-Roberta-snli-multinli-stsb",
                help="Name of a custom sentence transformer checkpoint.")
    parser.add_argument("--separator", default=",",
                help="Data field separator, e.g., ',' in csv files.")
    parser.add_argument("--seed", type=int, default=0,
                help='Seed to use to grant reproducibility of the training process of the SVM model.')
    parser.add_argument("-s", "--similarity_measure", default="cos", choices=["dot", "cos", "euclidean"],
                help="Default similarity score function")
    parser.add_argument("-t", "--topk", type=int, default=10,
                help="Top k documents of the rank to show in CLI")


    # Parse arguments.
    args = parser.parse_args()
    
    data = RankingDataset(datapath=args.datapath_documents,
                          datapath_queries=args.datapath_queries, 
                          datapath_query_doc=args.datapath_query_doc,
                          separator=args.separator,
                          sentence_transformer=args.sentence_transformer,
                          similarity_measure=args.similarity_measure
                          )
                          
    rankingModel = RankSVM(data, seed=args.seed)

    while (query := input("Query (type q to quit): ").strip().lower()) != 'q':
        rankingModel.show_ranking(query, top=args.t)