"""
Custom opt_tools tasks. Eventually, I think these can be merged into the main package.
"""
import time

import numpy as np
import os
import skimage.io as io
from skimage.transform import resize
import pandas as pd

import opt_tools


class GPflowBenchmarkTrackerBase(opt_tools.tasks.GPflowLogOptimisation):
    def __init__(self, test_X, test_Y, size_classes, sequence, trigger="iter", old_hist=None, store_fullg=False, store_x=None,
                 store_x_columns=None, verbose=False, test_data=0):
        opt_tools.tasks.GPflowLogOptimisation.__init__(self, sequence, trigger, old_hist, store_fullg, store_x,
                                                       store_x_columns)
        self.test_X = test_X
        self.test_Y = test_Y
        self.size_classes = size_classes
        self.verbose = verbose

        self.test_data_bool = test_data

        if test_data == 1:
            self.test_data = self.get_dogs_test_data()
        elif test_data == 2:
            self.test_data = self.get_mnist_test_data()

    def _get_record(self, logger, x, f=None):
        log_dict = super(GPflowBenchmarkTrackerBase, self)._get_record(logger, x, f)
        log_dict.update(logger.model.get_optimizer_variables()[0])
        return log_dict

    def get_dogs_test_data(self):
        data_path = '/Users/rasulkh/.kaggle/competitions/dog-breed-identification/test/'
        img_test_names = os.listdir(data_path)[:50]
        imgs_test = []

        for j, img in enumerate(img_test_names):
            img_path = os.path.join(data_path, img)
            imgs_test.append(resize(io.imread(img_path, as_grey=True), (200, 200), mode='reflect'))
            
        imgs_test = np.array(imgs_test, dtype=float).reshape(j+1, -1)
        print('Test data uploaded')

        return imgs_test

    def get_mnist_test_data(self):
        data = pd.read_csv('/Users/rasulkh/.kaggle/competitions/digit-recognizer/test.csv')
        return data.values / 255.0

class GPflowRegressionTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowRegressionTracker, self)._get_columns(logger) + ['rmse', 'nlpp', 'pred_time']

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowRegressionTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)


        pY, pYv = logger.model.predict_y(self.test_X)
        rmse = np.mean((pY - self.test_Y) ** 2.0) ** 0.5
        nlpp = -np.mean(-0.5 * np.log(2 * np.pi * pYv) - 0.5 * (self.test_Y - pY) ** 2.0 / pYv)
        log_dict.update({'rmse': rmse, 'nlpp': nlpp, 'pred_time': time.time() - st})

        if self.verbose:
            print("Benchmarks took %.2fs." % (time.time() - st))

        return log_dict


class GPflowBinClassTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowBinClassTracker, self)._get_columns(logger) + ['acc', 'nlpp']

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowBinClassTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)

        p, var = logger.model.predict_y(self.test_X)
        acc = ((p > 0.5).astype('float') == self.test_Y).mean()
        nlpp = -np.mean(np.log(p) * self.test_Y + np.log(1 - p) * (1 - self.test_Y))
        log_dict.update({'acc': acc, 'err': 1 - acc, 'nlpp': nlpp})

        if self.verbose:
            print("Benchmarks took %.2fs (err: %.4f, nlpp: %.4f)." % (time.time() - st, 1 - acc, nlpp))

        return log_dict


class GPflowMultiClassificationTracker(GPflowBenchmarkTrackerBase):
    def _get_columns(self, logger):
        return super(GPflowMultiClassificationTracker, self)._get_columns(logger) + ['acc', 'nlpp']

    def save_prediction(self, data):
        if self.test_data_bool == 1:
            labels = pd.read_csv('/Users/rasulkh/.kaggle/competitions/dog-breed-identification/labels.csv');
            indices = labels.breed.value_counts().index
            indices = indices[:self.size_classes]

            data_path = '/Users/rasulkh/.kaggle/competitions/dog-breed-identification/test/'
            img_test_names = os.listdir(data_path)[:50]
            img_test_names_new = [name.split('.')[0] for name in img_test_names]

            df = pd.DataFrame(data, index=img_test_names_new, columns=indices.values)
            df = df.loc[:, df.columns.sort_values()]
            df.reset_index(inplace=True)
            df.columns.values[0] = 'id'
            #df.sort_values('id').reset_index().iloc[:, 1:]
            df.to_csv('submission_dogs.csv', index=False)
        elif self.test_data_bool == 2:
            prediction = np.argmax(data, 1)
            df = pd.DataFrame(prediction, index=np.arange(1, 28001))
            df.reset_index(inplace=True)
            df.columns = ['ImageId', 'Label']
            df.to_csv('submission_mnist.csv', index=False)

        print('Predictions saved')

    def _get_record(self, logger, x, f=None):
        st = time.time()
        log_dict = super(GPflowMultiClassificationTracker, self)._get_record(logger, x, f)
        logger.model.set_state(x)

        pred_batch_size = 1000 if not hasattr(self, "pred_batch_size") else self.pred_batch_size

        p = np.vstack([logger.model.predict_y(self.test_X[n * pred_batch_size:(n + 1) * pred_batch_size, :])[0]
                       for n in range(-(-len(self.test_X) // pred_batch_size))])

        if self.test_data_bool != 0:
            p_test = np.vstack([logger.model.predict_y(self.test_data[n * pred_batch_size:(n + 1) * pred_batch_size, :])[0]
                           for n in range(-(-len(self.test_data) // pred_batch_size))])

            self.save_prediction(p_test)

        assert len(p) == len(self.test_X)
        # acc = ((p > 0.5).astype('float') == self.test_Y).mean()
        acc = np.array(np.argmax(p, 1) == self.test_Y[:, 0]).mean()
        pcorrect = p[self.test_Y == np.arange(0, self.size_classes)[None, :]]
        nlpp = -np.mean(np.log(pcorrect))

        log_dict.update({'acc': acc, 'err': 1 - acc, 'nlpp': nlpp})

        if self.verbose:
            print("Benchmarks took %.2fs (err: %.4f, nlpp: %.4f)." % (time.time() - st, 1 - acc, nlpp))

        return log_dict
