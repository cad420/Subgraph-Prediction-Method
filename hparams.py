import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # training scheme
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    # model
    parser.add_argument('--d_model', default=128, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=1024, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=20, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=20, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.5, type=float)

    # test
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")

    ###################################################################################################

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,help='Number of dimensions. Default is 128.')

    parser.add_argument('--dataset', default='ia-facebook-wall-wosn-dir', help='The name of dataset.')
    parser.add_argument('--node_num', type=int, default=46960, help='The number of node.')
    parser.add_argument('--max_graph_size', type=int, default=5, help='The max number of subgraph size.')
    parser.add_argument('--max_each', type=int, default=20, help='The number of subgraph samples from each node.')

    parser.add_argument('--split_r', default=0.5, type=float,
                        help="the rate of splitting data")
    parser.add_argument('--walkers', type=int, default=8, help='The number of walkers.')
    parser.add_argument('--spliter', type=int, default=8, help='The number of spliter.')
    parser.add_argument('--timestep', type=int, default=3, help='How far to transfer.')
    parser.add_argument('--type', default='dynamic', help='The type of training, {static,dynamic,attribute}')
