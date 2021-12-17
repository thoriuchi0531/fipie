class ReprMixin:

    def __repr__(self):
        """ Display class name and its parameters """
        cls = self.__class__.__name__
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{cls}({params})'
