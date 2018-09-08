"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    result = {}
    for byte in text:
        if byte not in result:
            result[byte] = 1
        else:
            result[byte] += 1
    return result


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    # source: https: // en.wikipedia.org / wiki / Huffman_coding
    # learned from wikipedia
    actual_dict = freq_dict
    if len(actual_dict) == 1:
        if list(freq_dict.keys())[0] == 255:
            actual_dict[254] = 0
        else:
            actual_dict[list(freq_dict.keys())[0] + 1] = 0
    freq_with_node = []
    for key in actual_dict:
        freq_with_node.append((actual_dict[key], HuffmanNode(key)))
    while len(freq_with_node) > 1:
        freq_with_node.sort()
        freq_with_node1 = freq_with_node[0]
        freq_with_node2 = freq_with_node[1]
        new_node = HuffmanNode(None, freq_with_node1[1], freq_with_node2[1])
        freq_with_node.remove(freq_with_node1)
        freq_with_node.remove(freq_with_node2)
        freq_with_node.append((freq_with_node1[0] + freq_with_node2[0],
                               new_node))
    return freq_with_node[0][1]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree2 = HuffmanNode(3)
    >>> e = get_codes(tree2)
    >>> e == {}
    True
    >>> tree3 = HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(None, HuffmanNode(6), HuffmanNode(7)), \
    HuffmanNode(5)), HuffmanNode(3))
    >>> f = get_codes(tree3)
    >>> f == {5: '01', 3: '1', 6: '000', 7: '001'}
    True
    """
    if tree is None:
        return {}
    elif tree.is_leaf():
        return {}
    else:
        return dict(get_codes_with_path(tree, ''))


def get_codes_with_path(tree, path):
    """ Return a list of tuples mapping symbols from <tree> rooted at
    HuffmanNode to codes according to the given <path>.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param str path: a start code-path for a node
    @rtype: list[tuple(int,str)]

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes_with_path(tree, '')
    >>> d == [(3, "0"), (2, "1")]
    True
    """
    result = []
    if tree.is_leaf():
        result.append((tree.symbol, path))
    else:
        left_side = get_codes_with_path(tree.left, path + '0')
        right_side = get_codes_with_path(tree.right, path + '1')
        result += left_side
        result += right_side
    return result


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    if tree is None:
        pass
    elif tree.is_leaf():
        pass
    else:
        number_nodes_with_num(tree, 0)


def number_nodes_with_num(tree, num):
    """ Number internal nodes in <tree> according to postorder traversal;
    start numbering at num. Return the next <num> for numbering.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param int num: the number to start numbering
    @rtype: int

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes_with_num(tree, 0)
    3
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # if t is a leaf, return the number, but don't change number property
    if tree.is_leaf():
        return num
    else:
        left_count = number_nodes_with_num(tree.left, num)
        right_count = number_nodes_with_num(tree.right, left_count)
        tree.number = right_count
        return tree.number + 1


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    code_tree = get_codes(tree)
    sum_ = 0
    total_freq = 0
    for key in code_tree:
        sum_ += len(code_tree[key]) * freq_dict[key]
        total_freq += freq_dict[key]
    return sum_ / total_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    result = []
    bit_str = ''
    for item in text:
        bit_str += codes[item]
    while len(bit_str) % 8 != 0:
        bit_str += '0'
    begin, end = 0, 8
    while end < len(bit_str) + 1:
        bits = bit_str[begin:end]
        result.append(bits_to_byte(bits))
        begin = end
        end += 8
    return bytes(result)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    if tree is None:
        return bytes([])
    elif tree.is_leaf():
        return bytes([])
    else:
        return bytes(act_on(tree, []))


def act_on(tree, container):
    """ Return a bytes representation of the tree rooted at tree into the
    container.

    @param HuffmanNode tree:
    @param list[int] container:
    @rtype: list[int]

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> act_on(tree, [])
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> act_on(tree, [])
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    if tree.is_leaf():
        return container
    elif tree.left.is_leaf() and tree.right.is_leaf():
        return container + [0, tree.left.symbol, 0, tree.right.symbol]
    elif tree.left.is_leaf():
        return act_on(tree.right, container) + [0, tree.left.symbol, 1,
                                                tree.right.number]
    elif tree.right.is_leaf():
        lst = act_on(tree.left, container)
        return lst + [1, tree.left.number, 0, tree.right.symbol]
    else:
        left = act_on(tree.left, container)
        right = act_on(tree.right, left)
        return right + [1, tree.left.number, 1, tree.right.number]


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    # tree_to_bytes fail if when len(freq) == 1, we add key 256
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    index_node = node_lst[root_index]
    left_check = index_node.l_type
    right_check = index_node.r_type
    if left_check == 0 and right_check == 0:
        return HuffmanNode(None, HuffmanNode(index_node.l_data),
                           HuffmanNode(index_node.r_data))
    elif right_check == 0 and left_check == 1:
        return HuffmanNode(None,
                           generate_tree_general(node_lst,
                                                 index_node.l_data),
                           HuffmanNode(index_node.r_data))
    elif right_check == 1 and left_check == 0:
        return HuffmanNode(None, HuffmanNode(index_node.l_data),
                           generate_tree_general(node_lst,
                                                 index_node.r_data))
    else:
        return HuffmanNode(None,
                           generate_tree_general(node_lst,
                                                 index_node.l_data),
                           generate_tree_general(node_lst,
                                                 index_node.r_data))


# def generate_tree_postorder(node_lst, root_index):
#     """ Return the root of the Huffman tree corresponding
#     to node_lst[root_index].
#
#     The function assumes that the list represents a tree in postorder.
#
#     @param list[ReadNode] node_lst: a list of ReadNode objects
#     @param int root_index: index in the node list
#     @rtype: HuffmanNode
#
#     >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
#     ReadNode(1, 0, 1, 0)]
#     >>> generate_tree_postorder(lst, 2)
#     HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
# HuffmanNode(7, None, None)), \
# HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
#     """
#     root = node_lst[root_index]
#     if root.l_type == 0 and root.r_type == 0:
#         return HuffmanNode(None, HuffmanNode(root.l_data, None, None),
#                            HuffmanNode(root.r_data, None, None))
#     elif root.l_type == 0 and root.r_type == 1:
#         return HuffmanNode(None, HuffmanNode(root.l_data, None, None),
#                            generate_tree_postorder(node_lst, root_index - 1))
#     elif root.l_type == 1 and root.l_type == 0:
#         return HuffmanNode(None, generate_tree_postorder(node_lst,
#                                                          root_index - 1),
#                            HuffmanNode(root.r_data, None, None))
#     else:
#         right_tree = generate_tree_postorder(node_lst, root_index - 1)
#         right_count = count_internal(right_tree)
#         left_tree = generate_tree_postorder(node_lst,
#                                             root_index - right_count - 1)
#         return HuffmanNode(None, left_tree, right_tree)


def generate_tree_post(node_lst, node_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.
    This means that we don't care about the first and third values in ReadNode.
    We explicitly number our nodes based on postorder traversal.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int node_index: index in the node list
    @rtype: HuffmanNode

    # Discussion:
    # With postorder traversal, we know the root node of a tree will always
    # be the last node in the list, so our strategy is to work backwards from
    # the end of the list. Any subtrees will exhibit the same property.
    #
    # This leads us to 2 very important insights:
    # 1. The node number of the right child will be 1 less than the root node.
    # 2. If the right child is not a leaf, then it will be a subtree with some
    # number of nodes > 0. We must traverse this subtree first before we can
    # move onto the left child. This means that the node number of the left
    # child will be the node number of the right child - the number of nodes
    # contained in the right subtree.
    """
    curr = node_lst[node_index]

    # Case #1: Neither child is a leaf. This means the previous node in our
    # read_node list corresponds to the root node of the right subtree.
    if curr.l_type == 1 and curr.r_type == 1:
        # Create our right subtree from the previous node_index
        t_right = generate_tree_post(node_lst, node_index - 1)

        # node number of the right subtree is the current node_index - 1
        t_right.number = node_index - 1

        # total number of nodes in right subtree
        t_right_number_of_nodes = count_internal(t_right)

        # set the node_index for the left_child
        t_left_node_index = node_index - t_right_number_of_nodes

        # recursively call our function on the left child.
        # the right child has already been traversed recursively,
        # so simply add it to the tree.
        tree = HuffmanNode(None,
                           generate_tree_post(node_lst, t_left_node_index - 1),
                           t_right)

        # node number of our parent tree is the current node_index
        tree.number = node_index

    # The rest of this code is analogous to the generate_tree_general() function
    elif curr.l_type == 1 and curr.r_type == 0:
        tree = HuffmanNode(None,
                           generate_tree_post(node_lst, node_index - 1),
                           HuffmanNode(curr.r_data))
        tree.number = node_index

    elif curr.l_type == 0 and curr.r_type == 1:
        tree = HuffmanNode(None,
                           HuffmanNode(curr.l_data),
                           generate_tree_post(node_lst, node_index - 1))
        tree.number = node_index
    else:
        tree = HuffmanNode(None,
                           HuffmanNode(curr.l_data),
                           HuffmanNode(curr.r_data))
        tree.number = node_index

    return tree


def count_internal(tree):
    """ Return the number of internal node of a HuffmanNode <tree>

    @param HuffmanNode tree:
    @rtype: int

    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1), \
HuffmanNode(2)), HuffmanNode(None, HuffmanNode(3), HuffmanNode(4)))
    >>> count_internal(t)
    3
    """
    if tree.is_leaf():
        return 0
    else:
        return 1 + count_internal(tree.left) + count_internal(tree.right)


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    bits_text = ""
    for byte in text:
        bits_text += byte_to_bits(byte)
    code_dict = get_codes(tree)
    code_to_symbol = {}
    for symbol in code_dict:
        code_to_symbol[code_dict[symbol]] = symbol
    begin, end, decode = 0, 1, []
    while end != (len(bits_text) + 1):
        code = bits_text[begin:end]
        if code in code_to_symbol:
            decode.append(code_to_symbol[code])
            begin = end
        end += 1
    return bytes(decode[:size])


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    freq_with_node = []
    for key in freq_dict:
        freq_with_node.append((freq_dict[key], key))
    freq_with_node.sort()

    def act(node):
        """ Assign the symbol of HuffmanNonde <node>

        @param HuffmanNode node:
        @rtype: None
        """
        if node.is_leaf():
            node.symbol = freq_with_node[-1][1]
            freq_with_node.pop()
    levelorder_visit(tree, act)


# source: lecture
def visit_level(t, n, act):
    """
    Visit each node of HuffmanNode t at level n and act on it.  Return
    the number of nodes visited visited.

    @param HuffmanNode|None t: HuffmanNode to visit
    @param int n: level to visit
    @param (HuffmanNode)->Any act: function to execute on nodes at level n
    @rtype: int

    >>> b1 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(6))
    >>> b2 = HuffmanNode(None, HuffmanNode(10), HuffmanNode(14))
    >>> b = HuffmanNode(8, b1, b2)
    >>> def f(node): print(node.symbol)
    >>> visit_level(b, 2, f)
    2
    6
    10
    14
    4
    """
    if t is None:
        return 0
    if n == 0:
        act(t)
        return 1
    elif n > 0:
        return (visit_level(t.left, n - 1, act)
                + visit_level(t.right, n - 1, act))
    else:
        return 0


# source: lecture
def levelorder_visit(t, act):
    """
    Visit HuffmanNode t in level order and act on each node.

    @param HuffmanNode|None t: HuffmanNode to visit
    @param (HuffmanNode)->Any act: function to use during visit
    @rtype: None

    >>> b1 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(6))
    >>> b2 = HuffmanNode(None, HuffmanNode(10), HuffmanNode(14))
    >>> b = HuffmanNode(None, b1, b2)
    >>> def f(node): print(node.symbol)
    >>> levelorder_visit(b, f)
    None
    None
    None
    2
    6
    10
    14
    """
    (visited, n) = (visit_level(t, 0, act), 0)
    while visited > 0:
        n += 1
        visited = visit_level(t, n, act)

if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
