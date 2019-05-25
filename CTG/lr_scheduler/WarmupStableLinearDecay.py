from torch.optim.lr_scheduler import _LRScheduler


class WarmupStableLinearDecay(_LRScheduler):

    def __init__(self, args, optimizer):
        self.lr = 1e-8
        super().__init__(optimizer)
        self.p = args['warmup_updates']
        self.s = args['start_decay']
        self.e = args['end_decay']
        self.max_lr = args['lr']
        self.min_lr = 0
        self.k1 = self.max_lr / self.p
        self.k2 = -self.max_lr / (self.e - self.s)

    def step(self, epoch=None):
        return self.lr

    def get_lr(self):
        return self.lr

    def step_update(self, num_updates):
        if num_updates < self.p:
            self.lr = self.k1 * num_updates
        elif num_updates < self.s:
            self.lr = self.max_lr
        elif num_updates < self.e:
            self.lr = self.max_lr + self.k2 * (num_updates - self.s)
        else:
            self.lr = 0