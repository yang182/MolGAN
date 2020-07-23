import argparse

args = argparse.ArgumentParser(description='MolGAN model for molecular.')
args.add_argument('--device', default=1)
args.add_argument("--mode", default="train", help='mode for model, default is true')
args.add_argument('--learning_rate', default=0.01, help='learning rate')
args.add_argument('--batch_dim', default=128, help='size of a batch')
args.add_argument('--la', default=0.9, type=float)
args.add_argument('--dropout_rate', default=0.5)
args.add_argument('--z_dim', default=8, type=int, help='the sample dim')
args.add_argument('--epochs', default=1000, type=int)
args.add_argument('--temperature', default=1.0, type=float)
args.add_argument('--n_critic', default=5, type=int, help='the ratio of train discriminator and generator' )

# parser = argparse.ArgumentParser(description='Process some integers.')
args.add_argument('--dataset', default='../data/', help='the data path')
# args.add_argument('--model', default='wavelet_gcn', help='model name')

args.add_argument('--max_atom_num', default=29)

args = args.parse_args()
print(args)


# batch_dim = 128
# la = 1
# dropout = 0
# n_critic = 5
# metric = 'validity,sas'
# n_samples = 5000
# z_dim = 8
# epochs = 10
# save_every = 1  # May lead to errors if left as None

