def readTree():
    Tree = {
        "value" : None,
        "left" : None,
        "right" : None
    }

    print("Node value:")
    value = input()
    Tree["value"] = value

    print("Has ", value, " left child? (1|0): ")
    has_child = input()
    if has_child == "1":
        Tree["left"] = readTree()

    print("Has ", value, " rigth child? (1|0): ")
    has_child = input()
    if has_child == "1":
        Tree["right"] = readTree()

    return Tree

def inOrder(Tree):
    result = []
    rigth = []

    if Tree["left"] != None:
        result = inOrder(Tree["left"])
    result.append(Tree["value"])
    if Tree["right"] != None:
        rigth = inOrder(Tree["right"])
        for i in rigth:
            result.insert(len(result), i)

    return result

root = readTree()
arbol = inOrder(root)

print(arbol)

