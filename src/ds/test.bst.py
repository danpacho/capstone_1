from bst import BST


def test_bst():
    # Create a BST
    bst = BST()

    # Insert keys into the BST
    bst.insert(5)
    bst.insert(3)
    bst.insert(7)
    bst.insert(2)
    bst.insert(4)
    bst.insert(6)
    bst.insert(8)
    bst.insert(5)  # Insert a duplicate key

    # Test search method
    assert bst.search(5).key == 5
    assert bst.search(10) is None

    assert bst.search(5).count == 2  # Key has duplicates
    # Test delete method
    bst.delete(5)  # Delete a key with duplicates
    assert bst.search(5).key == 5  # Key still exists in the BST
    bst.delete(10)  # Delete a non-existent key
    assert bst.search(10) is None  # Key does not exist in the BST

    # Test list property
    assert bst.list == [2, 3, 4, 5, 6, 7, 8]

    # Test insert_list method
    bst.insert_list([1, 9, 10])

    assert bst.search(1).key == 1
    assert bst.search(9).key == 9
    assert bst.search(10).key == 10

    # Test list property
    assert bst.list == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Test delete_list method
    bst.delete_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert bst.list == []

    print("BST test passed.")


test_bst()
