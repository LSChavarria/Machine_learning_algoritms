class Tree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.rigth = None
    
    def getLeft(self):
        return self.left

    def getRigth(self):
        return self.rigth

    def getValue(self):
        return self.value
    
    def setLeft(self, left):
        self.left = left

    def setRigth(self, rigth):
        self.rigth = rigth

def readTree():
    print("Node value:")
    value = input()
    tree = Tree(value)

    print("Has ", value, " left child? (1|0): ")
    has_child = input()
    if has_child == "1":
        tree.setLeft(readTree())

    print("Has ", value, " rigth child? (1|0): ")
    has_child = input()
    if has_child == "1":
        tree.setRigth(readTree())

    return tree

def inOrder(tree):
    result = []
    rigth = []

    if tree.getLeft() != None:
        result = inOrder(tree.getLeft())
    result.append(tree.getValue())
    if tree.getRigth() != None:
        rigth = inOrder(tree.getRigth())
        for i in rigth:
            result.insert(len(result), i)

    return result

root = readTree()
arbol = inOrder(root)

print(arbol)

