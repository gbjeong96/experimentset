import os

class Father():
    def __init__(self):
        self.something = 1

    def somefunction(self, X):
        print("Hello", X)


def main():
    father = Father()
    print(father)
    pred = father.somefunction(10)

if __name__ == "__main__":
    main()
