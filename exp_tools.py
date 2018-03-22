import contextlib
import os
import skimage.io as io
from skimage.transform import resize
import sys

import jug
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split

import GPflow
import GPflow.minibatch as mb
import opt_tools


@jug.TaskGenerator
def jugrun_experiment(exp):
    print("Running %s..." % exp.experiment_name)
    exp.setup()
    try:
        exp.run()
    except opt_tools.OptimisationTimeout:
        print("Timeout")


@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def load_mnist():
    data = pd.read_csv('/Users/rasulkh/.kaggle/competitions/digit-recognizer/train.csv')
    data_X = data.iloc[:, 1:].values / 255.0
    target = np.array(data.iloc[:, 0].values, dtype=int).reshape(-1, 1)
    X, Xt, Y, Yt = train_test_split(data_X, target, test_size=0.05)

    return X, Y, Xt, Yt

def load_fruits():
    imgs = []    
    target = []
    data_path = '/Users/rasulkh/.kaggle/competitions/fruits-360/Training'
    fruit_names = os.listdir(data_path)
    fruit_names.remove('.DS_Store')

    end_of_array = 0

    for i, fruit in enumerate(fruit_names):
        img_path = os.path.join(data_path, fruit)
        images = os.listdir(img_path)
        size_images = len(images)
        target[end_of_array : end_of_array + size_images] = i * np.ones(size_images)
        end_of_array += size_images + 1
        for img in images:    
            imgs.append(io.imread(os.path.join(img_path, img))[::4, ::4] / 255.0)

    imgs = np.array(imgs, dtype=float).reshape(len(target), -1)
    target = np.array(target, dtype=int).reshape(-1,1)

    X, Xt, Y, Yt = train_test_split(imgs, target)

    return X, Y, Xt, Yt

def load_dogs():
    labels = pd.read_csv('/Users/rasulkh/.kaggle/competitions/dog-breed-identification/labels.csv');
    indices = labels.breed.value_counts().index
    indices = indices[:3]
    new_labels = labels.set_index('breed').loc[indices].reset_index()

    imgs = []    
    target = []
    data_path = '/Users/rasulkh/.kaggle/competitions/dog-breed-identification/train/'

    dict_classes = dict()
    for i, name in enumerate(indices.values):
        dict_classes[name] = i

    for i, img in (new_labels.values):
        img_path = os.path.join(data_path, img) + '.jpg'
        imgs.append(resize(io.imread(img_path, as_grey=True), (200, 200), mode='reflect'))
        target.append(dict_classes[i])

    print(imgs[0])
    print(imgs[0].max(), imgs[0].min())
    imgs = np.array(imgs, dtype=float).reshape(len(target), -1)
    target = np.array(target, dtype=int).reshape(-1,1)

    print('Train data uploaded with ', len(imgs), ' images and ', len(np.unique(target)), ' classes')

    X, Xt, Y, Yt = train_test_split(imgs, target, test_size=0.15)
    print(Y)

    return X, Y, Xt, Yt

class ExperimentBase(object):
    def __init__(self, name):
        self.experiment_name = name
        self.m = None
        self.logger = None
        self.X = None
        self.Y = None
        self.Xt = None
        self.Yt = None
        self.size_classes = None
        self.run_settings = {}

    def setup_dataset(self, verbose=False):
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def setup_logger(self, verbose=False):
        raise NotImplementedError

    def setup(self, verbose=False):
        """
        setup
        Setup logger, model and anything else that isn't picklable.
        :return:
        """
        self.setup_dataset(verbose)
        self.setup_model()
        self.setup_logger(verbose)
        return self.m, self.logger

    def run(self, maxiter=np.inf):
        optimiser = self.run_settings.get("optimiser", "adam")
        if optimiser == "adam":
            opt_method = tf.train.AdamOptimizer(self.run_settings['learning_rate'])
        elif optimiser == "rmsprop":
            opt_method = tf.train.RMSPropOptimizer(self.run_settings['learning_rate'])
        else:
            opt_method = optimiser

        self.opt_method = opt_method

        try:
            return self.logger.optimize(method=opt_method, maxiter=maxiter, opt_options=self.run_settings)
        finally:
            self.logger.finish(self.m.get_free_state())

    def profile(self):
        """
        profile
        Run a few iterations and dump the timeline.
        :return:
        """
        s = GPflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_file_name = "./trace_" + self.experiment_name
        with GPflow.settings.temp_settings(s):
            self.m._compile()
            self.m._objective(self.m.get_free_state())
            self.m._objective(self.m.get_free_state())
            self.m._objective(self.m.get_free_state())

    def load_results(self):
        return pd.read_pickle(self.hist_path)

    @property
    def base_filename(self):
        return os.path.join('.', 'results', self.experiment_name)

    @property
    def hist_path(self):
        return self.base_filename + '_hist.pkl'

    @property
    def param_path(self):
        return self.base_filename + '_params.pkl'

    def __jug_hash__(self):
        from jug.hash import hash_one
        return hash_one(self.experiment_name)


class CifarExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        d = np.load('./datasets/cifar10.npz')
        self.X = (d['X'] / 255.0).reshape(50000, 3, 32, 32).swapaxes(1, 3).reshape(50000, -1)
        self.Y = d['Y'].astype('int64')
        self.Xt = (d['Xt'] / 255.0).reshape(10000, 3, 32, 32).swapaxes(1, 3).reshape(10000, -1)
        self.Yt = d['Yt'].astype('int64')

    def img_plot(self, i):
        import matplotlib.pyplot as plt
        plt.imshow(self.X[i, :].reshape(32, 32, 3))


class MnistExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        #with suppress_print():
        self.X, self.Y, self.Xt, self.Yt = load_mnist()
        self.size_classes=10

        print(self.Xt.shape, self.Yt.shape)


class FruitExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        self.X, self.Y, self.Xt, self.Yt = load_fruits()
        self.size_classes = len(np.unique(self.Y))
        print(self.size_classes)

class DogExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        self.X, self.Y, self.Xt, self.Yt = load_dogs()
        self.size_classes = len(np.unique(self.Y))
        print(self.size_classes)


class RectanglesImageExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        d = np.load('datasets/rectangles_im.npz')
        self.X, self.Y, self.Xt, self.Yt = d['X'], d['Y'], d['Xtest'], d['Ytest']
        self.size_classes = len(np.unique(self.Y))


def calculate_large_batch_lml(m, minibatch_size, batches, progress=False):
    """
    This does not work properly yet, presumably because it changes the state (w.r.t. _parent) of the model.
    """
    assert type(batches) == int, "`batches` must be an integer."
    old_mbX = m.X
    old_mbY = m.Y
    m.X = mb.MinibatchData(m.X.value, minibatch_size,
                           batch_manager=mb.SequenceIndices(minibatch_size, m.X.value.shape[0]))
    m.Y = mb.MinibatchData(m.Y.value, minibatch_size,
                           batch_manager=mb.SequenceIndices(minibatch_size, m.X.value.shape[0]))
    m._kill_autoflow()

    batch_lmls = []
    if progress:
        from tqdm import tqdm
        for _ in tqdm(range(batches)):
            batch_lmls.append(m.compute_log_likelihood())
    else:
        for _ in range(batches):
            batch_lmls.append(m.compute_log_likelihood())

    m.X = old_mbX
    m.Y = old_mbY

    m._kill_autoflow()

    import gc
    gc.collect()

    return np.mean(batch_lmls)


class CalculateFullLMLMixin(object):
    def _get_record(self, logger, x, f=None):
        log_dict = super(CalculateFullLMLMixin, self)._get_record(logger, x, f)
        model = logger.model
        minibatch_size = logger.model.X.index_manager.minibatch_size
        lml = calculate_large_batch_lml(model, minibatch_size, model.X.shape[0] // minibatch_size, True)
        print("full lml: %f" % lml)
        log_dict.update({"lml": lml})
        return log_dict


class GPflowMultiClassificationTrackerLml(CalculateFullLMLMixin,
                                          opt_tools.gpflow_tasks.GPflowMultiClassificationTracker):
    pass


class GPflowTrackLml(opt_tools.tasks.GPflowLogOptimisation):
    def _get_record(self, logger, x, f=None):
        model = logger.model
        minibatch_size = logger.model.X.index_manager.minibatch_size
        lml = calculate_large_batch_lml(model, minibatch_size, model.X.shape[0] // minibatch_size, True)
        print("full lml: %f" % lml)
        return {"i": logger._i, "t": logger.model.num_fevals, "t": logger._opt_timer.elapsed_time,
                "tt": logger._total_timer.elapsed_time, "lml": lml}
