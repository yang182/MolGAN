from utils.sparse_molecular_dataset import SparseMolecularDataset
from optimizers.gan import train_step
from models.model import MolGan
from config import args

data = SparseMolecularDataset()
data.load('data/gdb9_9nodes.sparsedataset')
model = MolGan(data.vertexes,
               data.bond_num_types,
               data.atom_num_types,
               args.z_dim,
               decoder_units=(128, 256, 512),
               discriminator_units=((128, 64), 128, (128, 64)),
               # decoder=decoder_adj,
               # discriminator=encoder_rgcn,
               soft_gumbel_softmax=False,
               hard_gumbel_softmax=False,
               batch_discriminator=True,
               batch=args.batch_dim, training=args.mode,
               learning_rate=args.learning_rate,
               dropout_rate=args.dropout_rate,
               temperature=args.temperature)


def test_model():
    pass


def train_model(epochs):
    n_critic = args.n_critic
    steps = len(data) // args.batch_dim
    for epoch in range(epochs + 1):
        for step in range(steps):
            if (steps * epoch + step) % n_critic == 0:
                train_g = True
            else:
                train_g = False
            loss = train_step(data, model, args.la, train_g, feature_matching=True, batch_dim=args.batch_dim)
            print("step {step}, loss: {loss[0]}, {loss[1]}, {loss[2]} ".format(step=step, loss=loss))
        model.save('data/model/molgan-{}.h5'.format(epoch))

def predict():
    pass


def main():

    if args.mode == 'train':
        train_model(args.epochs)
    elif args.mode == 'test':
        test_model()
    elif args.mode == 'predict':
        predict()
    else:
        raise ValueError


if __name__ == '__main__':
    main()
