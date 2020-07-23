import os
import numpy as np
from utils.sparse_molecular_dataset import SparseMolecularDataset
from optimizers.gan import train_step
from models.model import MolGan
from config import args
from utils.utils import mols2grid_image

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
            if step%50 == 0:
                predict("step-{}.bmp".format(step))
        model.save('data/model/molgan-{}.h5'.format(epoch))


def predict(figure_name):
    nodes_gumbel_argmax, edges_gumbel_argmax = model(inputs=None, training=False)
    n, e = nodes_gumbel_argmax, edges_gumbel_argmax
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    gen_mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
    smiles_figure = mols2grid_image(gen_mols, 3)
    figure_path = os.path.join('data/figure/', figure_name)
    smiles_figure.save(figure_path)


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
