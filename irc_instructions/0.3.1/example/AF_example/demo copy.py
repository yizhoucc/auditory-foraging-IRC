b = 20
def square(x):
    return x**2
class Nonsense():
    def __init__(self, x = None):
        print(square(x))

dummy_class = Nonsense(2)

