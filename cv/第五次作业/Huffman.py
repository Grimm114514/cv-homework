import heapq
from collections import defaultdict

# 定义节点类，用于构建哈夫曼树
class Node:
    def __init__(self, char, freq):
        self.char = char  # 字符
        self.freq = freq  # 字符出现的频率
        self.left = None  # 左子节点
        self.right = None  # 右子节点

    # 定义节点的比较规则，频率小的节点优先
    def __lt__(self, other):
        return self.freq < other.freq

# 计算每个字符的频率
# 输入：字符串数据
# 输出：字符频率的字典
def calculate_frequency(data):
    frequency = defaultdict(int)
    for char in data:
        frequency[char] += 1
    return frequency

# 构建哈夫曼树
# 输入：字符频率的字典
# 输出：哈夫曼树的根节点
def build_huffman_tree(frequency):
    # 将每个字符和频率封装成节点，并加入优先队列
    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)  # 将列表转化为最小堆

    # 合并频率最小的两个节点，直到队列中只剩一个节点（根节点）
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)  # 弹出频率最小的节点
        right = heapq.heappop(priority_queue)  # 弹出频率次小的节点
        merged = Node(None, left.freq + right.freq)  # 合并节点
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)  # 将合并后的节点重新加入队列

    return priority_queue[0]

# 构建哈夫曼编码表
# 输入：哈夫曼树的根节点，当前编码，编码表
# 输出：编码表（字符到二进制编码的映射）
def build_codes(node, current_code, codes):
    if node is None:
        return

    # 如果是叶子节点，记录字符对应的编码
    if node.char is not None:
        codes[node.char] = current_code
        return

    # 递归构建左子树和右子树的编码
    build_codes(node.left, current_code + "0", codes)
    build_codes(node.right, current_code + "1", codes)

# 哈夫曼编码主函数
# 输入：字符串数据
# 输出：编码后的数据和编码表
def huffman_encoding(data):
    frequency = calculate_frequency(data)  # 计算字符频率
    root = build_huffman_tree(frequency)  # 构建哈夫曼树
    codes = {}
    build_codes(root, "", codes)  # 构建编码表

    # 将数据编码为二进制字符串
    encoded_data = "".join(codes[char] for char in data)
    return encoded_data, codes

# 哈夫曼解码函数
# 输入：编码后的数据，哈夫曼树的根节点
# 输出：解码后的原始数据
def huffman_decoding(encoded_data, root):
    decoded_data = ""
    current_node = root
    for bit in encoded_data:
        # 根据当前位移动到左子树或右子树
        current_node = current_node.left if bit == "0" else current_node.right
        if current_node.char is not None:  # 如果到达叶子节点，记录字符
            decoded_data += current_node.char
            current_node = root  # 回到根节点

    return decoded_data

# 保存压缩后的文件
# 输入：编码后的数据，编码表，输出文件名
def save_compressed_file(encoded_data, codes, output_file):
    with open(output_file, "wb") as f:
        # 保存编码表（以字符串形式保存）
        f.write((str(codes) + "\n").encode())
        # 保存压缩数据
        f.write(encoded_data.encode())

# 压缩文件主函数
# 输入：原始文件名，输出文件名
def compress_file(input_file, output_file):
    with open(input_file, "rb") as f:
        data = f.read()  # 读取文件内容

    encoded_data, codes = huffman_encoding(data)  # 进行哈夫曼编码
    save_compressed_file(encoded_data, codes, output_file)  # 保存压缩后的文件

if __name__ == "__main__":
    input_file = '1.png'  # 输入文件名
    output_file = '1.huff'  # 输出文件名
    compress_file(input_file, output_file)  # 调用压缩函数
    print(f"文件 {input_file} 已成功压缩为 {output_file}")  # 打印成功信息
