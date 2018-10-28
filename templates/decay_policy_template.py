import abc

class Template_DecayPolicy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, optimizer, initial_learning_rate, iteration_wise):
        """
            optimizer: torch.optim.*
                Pytorch optimizer

            initial_learning_rate: float
                initial value of the learning rate.
                this will be the base value of each policy to calculate a next learning rate

            iteration_wise: bool
                Variable for using in the step-wise decaying or epoch-wise.
                So, it doesn't actually effect the codes, it's for management.
                Default value of this depends on a inherited class.
        """

        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.iteration_wise = iteration_wise

    @abc.abstractmethod
    def decay_lr(self, **kwargs):
        raise NotImplementedError()
