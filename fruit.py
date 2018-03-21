"""
Run the convolutional GP on Fruits, using multi classification
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import GPflow
import exp_tools
import opt_tools
import convgp.convkernels as ckern


class FruitExperiment(exp_tools.FruitExperiment):
    def __init__(self, name=None, M=100, run_settings=None):
        name = "fruit-%s%i" % (run_settings['kernel'], M) if name is None else name
        super(FruitExperiment, self).__init__(name)
        self.run_settings = run_settings if run_settings is not None else {}
        self.M = M

    def setup_model(self):
        Z = None
        if self.run_settings['kernel'] == "rbf":
            k = GPflow.kernels.RBF(25 * 25 * 3)
            Z = self.X[np.random.permutation(len(self.X))[:self.M], :]
        # elif self.run_settings['kernel'] == "poly":
        #     k = GPflow.kernels.Polynomial(25 * 25, degree=9.0)
        #     Z = self.X[np.random.permutation(len(self.X))[:self.M], :]
        # elif self.run_settings['kernel'] == "rbfpoly":
        #     k = (GPflow.kernels.Polynomial(25 * 25, degree=9.0) + GPflow.kernels.RBF(25 * 25) +
        #          GPflow.kernels.White(25 * 25, 1e-1))
        #     Z = self.X[np.random.permutation(len(self.X))[:self.M], :]
        elif self.run_settings["kernel"] == "conv":
            k = ckern.ConvRBF([25, 25], [5, 5]) + GPflow.kernels.White(1, 1e-3)
        elif self.run_settings['kernel'] == "wconv":
            k = ckern.WeightedConv(GPflow.kernels.RBF(25), [25, 25], [5, 5], colour_channels=3) + GPflow.kernels.White(1, 1e-3)
        else:
            raise NotImplementedError

        if Z is None:
            Z = (k.kern_list[0].init_inducing(self.X, self.M, method=self.run_settings['Zinit'])
                 if type(k) is GPflow.kernels.Add else
                 k.init_inducing(self.X, self.M, method=self.run_settings['Zinit']))

        k.fixed = self.run_settings.get('fixed', False)

        self.m = GPflow.svgp.SVGP(self.X, self.Y, k, GPflow.likelihoods.MultiClass(self.size_classes), Z.copy(), num_latent=self.size_classes,
                                  minibatch_size=self.run_settings.get('minibatch_size', self.M))
        if self.run_settings["fix_w"]:
            self.m.kern.W.fixed = True

    def setup_logger(self, verbose=False):
        h = pd.read_pickle(self.hist_path) if os.path.exists(self.hist_path) else None
        if h is not None:
            print("Resuming from %s..." % self.hist_path)
        tasks = [
            opt_tools.tasks.DisplayOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            opt_tools.tasks.GPflowLogOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            exp_tools.GPflowMultiClassificationTrackerLml(
                self.Xt[:, :], self.Yt[:, :], self.size_classes, itertools.count(1800, 1800), trigger="time",
                verbose=True, store_x="final_only", store_x_columns=".*(variance|lengthscales)"),
            opt_tools.gpflow_tasks.GPflowMultiClassificationTracker(
                self.Xt[:, :], self.Yt[:, :], self.size_classes, opt_tools.seq_exp_lin(1.5, 150, start_jump=30), trigger="time",
                verbose=True, store_x="final_only", store_x_columns=".*(variance|lengthscales)", old_hist=h),
            opt_tools.tasks.StoreOptimisationHistory(self.hist_path, opt_tools.seq_exp_lin(1.5, 600, start_jump=30),
                                                     trigger="time", verbose=False),
            opt_tools.tasks.Timeout(self.run_settings.get("timeout", np.inf))
        ]
        self.logger = opt_tools.GPflowOptimisationHelper(self.m, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MNIST experiment.')
    parser.add_argument('--fixed', '-f', help="Fix the model hyperparameters.", action="store_true", default=False)
    parser.add_argument('--name', '-n', help="Experiment name appendage.", type=str, default=None)
    parser.add_argument('--learning-rate', '-l', help="Learning rate.", type=str, default="0.001")
    parser.add_argument('--learning-rate-block-iters', type=int, default=50000,
                        help="How many iterations to use in a run with a single learning rate.")
    parser.add_argument('--profile', help="Only run a quick profile of an iteration.", action="store_true",
                        default=False)
    parser.add_argument('--no-opt', help="Do not optimise.", action="store_true", default=False)
    parser.add_argument('-M', help="Number of inducing points.", type=int, default=100)
    parser.add_argument('--minibatch-size', help="Size of the minibatch.", type=int, default=100)
    parser.add_argument('--benchmarks', action="store_true", default=False)
    parser.add_argument('--optimiser', '-o', type=str, default="adam")
    parser.add_argument('--fix-w', action="store_true", default=False)
    parser.add_argument('--kernel', '-k', help="Kernel.")
    parser.add_argument('--Zinit', help="Inducing patches init.", default="patches-unique", type=str)
    parser.add_argument('--lml', help="Compute log marginal likelihood.", default=False, action="store_true")
    args = parser.parse_args()

    # if GPflow.settings.dtypes.float_type is not tf.float32:
    #     raise RuntimeError("float_type must be float32, as set in gpflowrc.")

    run_settings = vars(args).copy()
    del run_settings['profile']
    del run_settings['no_opt']
    del run_settings['name']
    exp = FruitExperiment(name=args.name, M=args.M, run_settings=run_settings)

    if args.profile:
        print("Profiling an iteration...")
        exp.setup()
        exp.profile()
    elif not args.no_opt:
        lr_base = run_settings['learning_rate']
        while True:
            print(exp.experiment_name)
            i = pd.read_pickle(exp.hist_path).i.max() if os.path.exists(exp.hist_path) else 1.0
            b = args.learning_rate_block_iters
            print("learning rate: %s" % args.learning_rate)
            run_settings['learning_rate'] = eval(args.learning_rate)  # Can use i and b in learning_rate
            print(run_settings['learning_rate'], i)
            exp.setup()
            rndstate = np.random.randint(0, 1e9)
            exp.m.X.index_manager.rng = np.random.RandomState(rndstate)
            exp.m.Y.index_manager.rng = np.random.RandomState(rndstate)
            exp.run(maxiter=args.learning_rate_block_iters)

    if args.benchmarks:
        exp.setup()
        p = exp.m.predict_y(exp.Xt)[0]
        lpp = np.mean(np.log(p[np.arange(len(exp.Xt)), exp.Yt.flatten()]))
        acc = np.mean(p.argmax(1) == exp.Yt[:, 0])

        print("Accuracy: %f" % acc)
        print("Lpp     : %f" % lpp)

    if args.lml:
        exp.setup()
        lml = exp_tools.calculate_large_batch_lml(exp.m, args.minibatch_size, exp.m.X.shape[0] // args.minibatch_size,
                                                  progress=True)
        print(lml)
