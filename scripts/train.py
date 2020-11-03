import os
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mnist_pl import MNISTDataModule, LitClassifier


def main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MNISTDataModule.add_data_specific_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = MNISTDataModule('', batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
