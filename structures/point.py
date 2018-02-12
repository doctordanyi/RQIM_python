class Point2D:
    def __init__(self, coordinates=(0, 0)):
        self.coordinates = coordinates

    @property
    def x(self):
        return self.coordinates[0]

    @x.setter
    def x(self, x):
        self.coordinates[0] = x

    @property
    def y(self):
        return self.coordinates[1]

    @y.setter
    def y(self, y):
        self.coordinates[0] = y

    def __str__(self):
        return "({},{})".format(self.x, self.y)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self))
