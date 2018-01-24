class Quad:

    def __init__(self):
        self.corners = [(), (), (), ()]


def test():
    quad = Quad()
    print(type(quad.corners[0]))


if __name__ == "__main__":
    test()
