import numpy as np
from torch.utils.data import Sampler
from .IC_dataset_new import IC_dataset

def worker_init_fn_seed(worker_id):
    seed = 10
    seed = seed + worker_id
    np.random.seed(seed)

class BalancedBatchSampler(Sampler):
    def __init__(self,
                 cfg,
                 dataset: IC_dataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        label_list = list()
        for i in self.dataset:
            label_list.append(int(i[2]))

        label_list = np.array(label_list)
        normal_idx = np.argwhere(label_list == 0).flatten()
        outlier_idx = np.argwhere(label_list == 1).flatten()

        self.normal_generator = self.randomGenerator(normal_idx)
        self.outlier_generator = self.randomGenerator(outlier_idx)

        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))

        self.n_normal = batch_size // 2
        self.n_outlier = batch_size - self.n_normal

        print(self.cfg.steps_per_epoch)

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.cfg.steps_per_epoch

    
    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                 batch.append(next(self.outlier_generator))

            yield batch
