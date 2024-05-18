from typing import Union


class Node:
    """
    Node with count for duplicated keys
    """

    def __init__(self, key: float) -> None:
        self.key = key
        self.left: Union[Node, None] = None
        self.right: Union[Node, None] = None
        self.count = 1

    def add_count(self):
        """
        Add the count of the node
        """
        self.count += 1

    def sub_count(self):
        """
        Subtract the count of the node
        """
        self.count -= 1


class BST:
    """
    Binary Search Tree (duplicates allowed)
    """

    def __init__(self) -> None:
        self.root = None

    def insert(self, key: float) -> None:
        """
        Insert a key into the BST
        """
        self.root = self._insert(self.root, key)

    def insert_list(self, key_list: list[float]) -> None:
        """
        Insert a list of keys into the BST
        """
        for key in key_list:
            self.insert(key)

    def _insert(self, root: Node, key: float) -> Node:
        if root is None:
            return Node(key)

        if key < root.key:
            root.left = self._insert(root.left, key)
        elif key > root.key:
            root.right = self._insert(root.right, key)
        else:
            root.add_count()
        return root

    def search(self, key: int) -> Union[Node, None]:
        """
        Search for a key in the BST
        """
        return self._search(self.root, key)

    def _search(self, root: Node, key: float) -> Union[Node, None]:
        if root is None:
            return None
        if root.key == key:
            return root

        if key < root.key:
            return self._search(root.left, key)
        return self._search(root.right, key)

    def delete(self, key: int) -> Union[Node, None]:
        """
        Delete a key from the BST
        """
        self.root = self._delete(self.root, key)
        return self.root

    def delete_list(self, key_list: list[float]) -> None:
        """
        Delete a list of keys from the BST
        """
        for key in key_list:
            self.delete(key)

    def _delete(self, root: Node, key: float):
        if root is None:
            return root
        if key < root.key:
            root.left = self._delete(root.left, key)
        elif key > root.key:
            root.right = self._delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left

            if root.count > 1:  # If the key has duplicates
                root.sub_count()
                return root

            root.key = self._find_successor(root.right)
            root.right = self._delete(root.right, root.key)

        return root

    def _find_successor(self, node) -> float:
        current = node
        while current.left is not None:
            current = current.left
        return current

    @property
    def list(self) -> list[float]:
        """
        Return the inorder traversal of the BST, list[float]
        """
        return self._inorder(self.root)

    def _inorder(self, root: Node):
        if root is None:
            return []
        return (
            self._inorder(root.left)
            + [root.key] * root.count
            + self._inorder(root.right)
        )

    def reset(self):
        """
        Reset the BST
        """
        self.root = None
