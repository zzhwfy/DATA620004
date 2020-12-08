import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import example
import copy
###################################################################
#棋盘：example.board
def evaluate_type():
    #衡量各种棋形的好坏
    score = {}
    score['5'] = 100000  # 连五：五颗同色的棋子连在一起，该方获胜
    score['4'] = 50000  # 活四：某一方有两个点位可以形成连五，这样第二方无论怎么堵该方均会获胜
    score['h4'] = 10000  # 冲四：某一方有一个点位可以形成连五
    score['3'] = 5000  # 活三：可以形成活四的三
    score['h3'] = 3000  # 眠三：只能形成冲四的三
    score['2'] = 2000  # 活二：可以形成活三的二
    score['h2'] = 1000  # 眠二：只能形成眠三的二
    return score

def dict_initialization():
    '''
    初始化棋形字典
    返回：
    一个字典，包括了各种有利棋形
    *有利棋形种类参考：https://blog.csdn.net/marble_xu/article/details/90450436
    '''
    godict = {}
    godict['5'] = 0
    godict['4'] = 0
    godict['h4'] = 0
    godict['3'] = 0
    godict['h3'] = 0
    godict['2'] = 0
    godict['h2'] = 0
    return godict

def extract_features(board, user = 1):
    '''
    从棋盘中获取对user有利的棋形，以便进一步评估
    参数：
    board：代表棋盘的二维数组(state)。board[i][j]=0/1/2
    user：代表己方/对手的视角
    返回：
    一个字典，包括了各种有利棋形以及出现次数
    '''
    godict = dict_initialization()
    direction = [(0,1), (1,0), (1,1), (1,-1)]
    mask = [[[0 for k in range(4)] for j in range(pp.height)] for i in range(pp.width)]
    #用mask来标记已经在该方向上计算过棋形的棋子，避免重复的同时减少计算量
    for i in range(pp.width):
        for j in range(pp.height):
            for k in range(4):
                if board[i][j] == user and mask[i][j][k] == 0:
                    direct = direction[k]
                    mask, godict = get_local_feature(board, mask, godict, user,i,j,k, direct) #在局部判断
    return godict

def get_localspace(board, i,j, user, direct, local_size = 4):
    '''
    按照方向获取局部信息，越界的部分用对手的棋子填充
    参数：
    board：代表棋盘的二维数组。board[i][j]=0/1/2
    i,j：当前的位置参数
    user：代表己方/对手
    direct：方向，(0,1), (1,0), (1,1), (1,-1)
    返回：
    一维数组localspace，记载了以i,j位置为中心，direct为（双边）方向上局部空间的落子信息
    *localspace[4]即为board[i][j]
    '''
    localspace = []
    op = 3-user #对手的代表符号
    first_i = i - local_size * direct[0]
    first_j = j - local_size * direct[1]
    for k in range(2*local_size + 1):
        temp_i = first_i + k * direct[0]
        temp_j = first_j + k * direct[1]
        if temp_i < 0 or temp_i >= pp.width or temp_j < 0 or temp_j >= pp.height:
            localspace.append(op)
        else:
            localspace.append(board[temp_i][temp_j])
    return localspace

def set_mask(mask, i, j, local_size, left, right, k, direct):
    '''
    将mask中已被探索过的部分设置为1
    参数：
    mask：标记棋盘上位置是否已被计算过的二维数组
    i：中心棋子的横坐标
    j：中心棋子的纵坐标
    local_size：局部空间的大小
    left：标记位置的最左边
    right：标记位置的最右边
    k： 方向的index
    direct：方向
    返回：
    更新后的mask
    '''
    first_i = i - local_size * direct[0]
    first_j = j - local_size * direct[1]
    for index in range(left, right+1):
        temp_i = first_i + index * direct[0]
        temp_j = first_j + index * direct[1]
        if temp_i >= 0 and temp_i < pp.width and temp_j >= 0 and temp_j < pp.height:
            mask[temp_i][temp_j][k] = 1
    return mask

def get_local_feature(board, mask, godict, user,i,j,k,direct):
    '''
    从棋盘中获取对user有利的棋形，以便进一步评估
    参数：
    board：代表棋盘状态的二维数组
    mask：标记棋盘上位置是否已被计算过的二维数组
    godict：记录有利棋形以及出现次数的字典
    user：代表己方/对手
    i,j：当前坐标
    k: 方向在方向数组中的位置，用于更新
    direct：方向
    返回：
    更新后的godict
    '''
    local_size = 4
    op = 3 - user #对手的编码
    localspace = get_localspace(board, i, j, user, direct, local_size)  # 以i,j位置为中心获取局部信息
    left = right = local_size
    #从中间位置的棋子开始向左/右逐步拓展，找寻连在一起的棋子范围
    while left > 0 and localspace[left-1] == user :
        left = left - 1
    while right < 2*local_size and localspace[right+1] == user:
        right = right + 1
    #找寻可行域范围：未被对手的棋阻断
    left_2 = left
    right_2 = right
    while left_2 > 0 and localspace[left_2-1] != op :
        left_2 = left_2 - 1
    while right_2 < 2*local_size and localspace[right_2+1] != op :
        right_2 = right_2 + 1
    #若可行域长度小于5，说明这块棋再怎么下都失去了连成5的可能性，不计算它的有利棋形
    if right_2+1-left_2 < 5:
        mask = set_mask(mask, i, j, local_size, left_2, right_2, k, direct)
    else:
        #可行域大于等于5的情况，根据连在一起的棋子个数right-left+1来分类讨论
        mask = set_mask(mask, i, j, local_size, left, right, k, direct) #先标记连在一起的部分
        length = right-left+1
        #五连！
        if length == 5 :
            godict['5'] = godict['5'] + 1
        #四连的情况，根据左右两边紧靠着的地方是否有对手的棋子来判断是活四还是冲四
        #活四：011110 冲四：211110/011112 （211112的情况之前已经筛掉了）
        if length == 4 :
            if localspace[left - 1] == op or localspace[right + 1] == op:
                godict['h4'] = godict['h4']+1
            else:
                godict['4'] = godict['4'] + 1
        #三连的情况，有可能是冲四/活三/眠三
        #冲四： 11101/10111（1011101的情况计两次冲四） 活三： 011100/001110 眠三： 21110/01112/2011102
        #需要往左/右分别再探索两步
        if length == 3:
            if localspace[left - 1] == op:
                if localspace[right + 2] == user:
                    mask = set_mask(mask, i, j, local_size, right + 1, right + 2, k, direct)
                    godict['h4'] = godict['h4'] + 1
                else:
                    godict['h3'] = godict['h3']+1
            if localspace[right + 1] == op:
                if localspace[left - 2] == user:
                    mask = set_mask(mask, i, j, local_size, left - 2, left - 1, k, direct)
                    godict['h4'] = godict['h4'] + 1
                else:
                    godict['h3'] = godict['h3']+1
            if localspace[left - 1] == 0 and localspace[right + 1] == 0: # ?01110?的情况
                if localspace[left - 2] == user: #10111
                    mask = set_mask(mask, i, j, local_size, left - 2, left - 1, k, direct)
                    godict['h4'] = godict['h4'] + 1
                if localspace[right + 2] == user: #11101
                    mask = set_mask(mask, i, j, local_size, right + 1, right+2, k, direct)
                    godict['h4'] = godict['h4']+1
                if localspace[left - 2] == 0 or localspace[right + 2] == 0: #001110/011100
                    godict['3'] = godict['3'] + 1
                if localspace[left - 2] == op and localspace[right + 2] == op: #2011102
                    godict['h3'] = godict['h3'] + 1
        #二连的情况。冲四/活三/眠三/活二/眠二
        #冲四： 11011，考虑到重复仅看右边的 活三： 011010/010110 （01011010的情况计两次活三）
        #眠三：211010/011012/210110/010112 活二：001102/201100/001100 眠二：21100/00112
        if length == 2:
            if localspace[right + 1] == 0 and localspace[right + 2] == user and localspace[right + 3] == user:  # 11011
                mask = set_mask(mask, i, j, local_size, right + 1, right + 3, k, direct)
                godict['h4'] = godict['h4'] + 1
            if localspace[right + 1] == 0 and localspace[left - 1] == 0: #??0110??
                if localspace[right + 2] == user and localspace[right + 3] == 0: #011010
                    mask = set_mask(mask, i, j, local_size, right + 2, right + 3, k, direct)
                    godict['3'] = godict['3'] + 1
                if localspace[right + 2] == user and localspace[right + 3] == op: #011012
                    mask = set_mask(mask, i, j, local_size, right + 2, right + 3, k, direct)
                    godict['h3'] = godict['h3'] + 1
                if localspace[left - 3] == 0 and localspace[left - 2] == user: #010110
                    mask = set_mask(mask, i, j, local_size, left - 3, left - 2, k, direct)
                    godict['3'] = godict['3'] + 1
                if localspace[left - 3] == op and localspace[left - 2] == user: #210110
                    mask = set_mask(mask, i, j, local_size, left - 3, left - 2, k, direct)
                    godict['h3'] = godict['h3'] + 1
                if localspace[right + 2] != user and localspace[left - 2] != user: # 001102/201100/001100
                    godict['2'] = godict['2'] + 1
            if localspace[right + 1] == op: # ??0112
                if localspace[left-2] == 0: # 00112
                    godict['h2'] = godict['h2'] + 1
                if localspace[left-2] == user and localspace[left-3] == 0: # 010112
                    mask = set_mask(mask, i, j, local_size, left - 3, left - 2, k, direct)
                    godict['h3'] = godict['h3'] + 1
            if localspace[left - 1] == op: # 2110??
                if localspace[right+2] == 0: # 21100
                    godict['h2'] = godict['h2'] + 1
                if localspace[right + 2] == user and localspace[right + 3] == 0: # 211010
                    mask = set_mask(mask, i, j, local_size, right + 2, right + 3, k, direct)
                    godict['h3'] = godict['h3'] + 1
        #孤子的情况。活二/眠二
        #活二： 01010/010010，考虑到重复仅计数右边的 眠二：21010/01012
        if length == 1:
            if localspace[right + 1] == op: # ??012
                if localspace[left - 2] == user and localspace[left - 3] == 0: # 01012
                    mask = set_mask(mask, i, j, local_size, left - 3, left - 2, k, direct)
                    godict['h2'] = godict['h2'] + 1
            elif localspace[left - 1] == op: # 210??
                if localspace[right + 2] == user and localspace[right + 3] == 0: # 21010
                    mask = set_mask(mask, i, j, local_size, right + 2, right + 3, k, direct)
                    godict['h2'] = godict['h2'] + 1
            else: # 010???
                if localspace[right + 2] == user and localspace[right + 3] == 0: # 01010
                    mask = set_mask(mask, i, j, local_size, right + 2, right + 3, k, direct)
                    godict['2'] = godict['2'] + 1
                if  localspace[right + 2] == 0 and localspace[right + 3] == user and localspace[right + 4] == 0: # 010010
                    mask = set_mask(mask, i, j, local_size, right + 2, right + 4, k, direct)
                    godict['2'] = godict['2'] + 1
    return mask, godict

####################################################################################

class Node:
    """
    属性:
        rule: 1 or 2，1代表MAX节点，2代表MIN节点
        successor: 当前节点的子节点列表
        board: 描述当前节点对应的棋盘状态
        is_leaf: bool，当前节点是否是叶节点
        value: 节点的效用值评分
        action: 一个元组，描述父节点到子节点的落子位置
    """
    def __init__(self, rule= 1, board=None,successor=None, is_leaf=False, value=None,action=None):
        if successor is None:
            successor = []
        self.rule = 'max' if rule == 1 else 'min'
        self.successor = successor
        self.board = copy.deepcopy(board)
        self.is_leaf = is_leaf
        self.value = value
        self.action = action
        '''self.godict_1 = extract_features(board, user = 1)
        self.godict_2 = extract_features(board, user = 2)'''

def evaluate(node):
    # 为节点进行评分的评分函数
    score = evaluate_type()
    discont = 1.1
    value = 0
    godict_1 = extract_features(node.board, user=1)
    godict_2 = extract_features(node.board, user=2)
    for key in godict_1:
        value = value + score[key]*godict_1[key] - discont*score[key]*godict_2[key]
    return value

#value_list = []

def get_value(node, alpha, beta):
    # 获得节点的评分值，先把节点依照类型进行分类
    if node.is_leaf == True:
        node.value=evaluate(node)
        #value_list.append(node.value)
        return node.value
    elif node.rule == 'max':
        return max_value(node,alpha,beta)
    elif node.rule == 'min':
        return min_value(node,alpha,beta)

def max_value(node, alpha, beta):
    """为MAX节点进行效用评分
    Args:
        node: class Node object
        alpha: float，为MAX节点传给其子节点的MAX目前能达到的最大值，在赋值过程中会被更新
        beta: float，为父节点传给MAX节点的父节点目前能达到的最小值，作为剪枝的标准
    """
    v= float("-inf")
    for child in node.successor:
        child.visited=1
        v=max(v,get_value(child,alpha,beta))
        alpha=max(alpha,v)
        if v>=beta:
            node.value=v
            return v
    node.value = v
    return v


def min_value(node, alpha, beta):
    """为MIN节点进行效用评分
    Args:
        node: class Node object
        alpha: float，为父节点传给MIN节点的父节点目前能达到的最大值，作为剪枝的标准
        beta: float，为MIN节点传给其子节点的MIN目前能达到的最小值，在赋值过程中会被更新
    """
    v = float("inf")
    for child in node.successor:
        child.visited = 1
        v = min(v, get_value(child, alpha, beta))
        beta = min(beta, v)
        if v <= alpha:
            node.value = v
            return v
    node.value = v
    return v

# def get_unvisited_nodes(node):
#     """Get unvisited nodes for the tree.
#
#     Args:
#         node: class Node object, root node of the current tree (or leaf)
#
#     Returns:
#         float list of values of the unvisited nodes.
#     """
#     unvisited = []
#     if node.successor:
#         for successor in node.successor:
#             unvisited += get_unvisited_nodes(successor)
#     else:
#         if not node.visited:
#             unvisited.append(node.value)
#     return unvisited


def hasNeighbor(board, x,y,radius):
    # 判断计划落子位置(x,y)附近(x+-radius,y+-radius)范围内是否有已经下好棋子
    indicator=0
    # 如果有表示这是在构建博弈树时我们会考察的落子位置，函数返回1
    for i in range(x-radius,x+radius+1):
        for j in range (y-radius,y+radius+1):
            if i>=0 and j>=0 and i<pp.width and j<pp.height and board[i][j]!=0:
                indicator = 1
                break
    return indicator

def generatesuccessor(node):
    # 读取当前节点的棋盘状态，生成它的子节点，返回子节点列表
    board = copy.deepcopy(node.board)
    # color表示落子的颜色，如果是MAX节点即当前状态是电脑走棋，则颜色为1
    color = 1 if node.rule=='max' else 2
    successorlist = []
    '''这里还需要判断node的棋盘状态是不是终止状态，如果是则返回[],并将node的is_leaf属性更新为True'''
    if node.is_leaf == True:
        return []
    for x in range(pp.width):
        for y in range(pp.height):
            if board[x][y]==0 and hasNeighbor(node.board,x,y,1):
                # 如果(x,y)是合法落子位置，则在此位置落子生成一个对应的子节点，子节点的rule与node相反
                board[x][y] = color
                child = Node(rule=3-color,board=board,successor=None,is_leaf=False,value=None,action=(x,y))
                '''if child.godict_1['5'] > 0:
                    child.is_leaf = True'''
                successorlist.append(child)
            board = copy.deepcopy(node.board) #board重置为父节点的棋盘状态
    node.successor = successorlist
    return successorlist

def construct_tree(depth_limit, rootnode):
    """从根节点开始构建一棵搜索树，每当轮到电脑走棋时，电脑读取当前棋盘状态并创建一个根节点
    从根节点开始进行深度有限搜索构造出一棵搜索树，并返回其根节点，每个节点都是一个Node对象
    参数：
        n: int, the height of tree
        tree: the input tree described with list nested structure
        rule: int, root node's type, 1 for max, 0 for min
    Hint: tree structure example
            属性:
        rule: 0 or 1，1代表MAX节点，0代表MIN节点
        successor: 当前节点的子节点列表
        board: 描述当前节点对应的棋盘状态
        is_leaf: bool，当前节点是否是叶节点
        value: 节点的效用值评分
        visited: bool, visited or not
        and each child has similar structure of root_node
    """
    depth = 0 # 初始深度为0
    generation = [rootnode] # 每轮需要生成子节点的节点列表，深度为0时只有根节点
    while depth<=depth_limit:
        if depth < depth_limit:
            newgeneration=[]# 每一轮开始时将列表清空
            while len(generation)!=0:
                node = generation.pop()
                for child in generatesuccessor(node):
                    newgeneration.append(child)# 将本轮所有生成的子节点加入列表
            generation = newgeneration
        if depth == depth_limit:
            while len(generation) != 0:
                node = generation.pop()
                newgeneration = []
                for child in generatesuccessor(node):
                    child.is_leaf = True  # 最后一轮所有的子节点都是叶节点
                    newgeneration.append(child)  # 将某个节点生成的子节点加入列表
        depth += 1
    return rootnode

def MiniMax(board):
    # 极大化极小博弈算法主函数，当轮到电脑落子时，读取棋盘信息，构造搜索树，给出落子策略
    currentboard = copy.deepcopy(board)# 当前棋盘信息
    # 当使用MIniMax状态评估时，说明轮到了电脑走，因此根节点一定是Max节点，使用这些信息构建根节点
    rootnode = Node(rule=1,board=currentboard,successor=None,is_leaf=False,value=None)
    # 生成用于状态效用评估的博弈树，gametree是其根节点
    gametree = construct_tree(depth_limit=0,rootnode=rootnode)
    # 为博弈树中的节点生成效用评分
    get_value(gametree, float("-inf"), float("inf"))
    max_utility = gametree.value # 根节点是max节点，它的效用值即是最优子节点的效用值
    max_child = rootnode
    for child in gametree.successor:
        if abs(child.value - max_utility) < 1e-4:
            max_child = child
            break
    return max_child.action, rootnode #返回从根节点到其最优子节点对应的策略



# def main():
#     while True:
#         try:
#             rule, n = map(int, input().strip().split())
#             tree = eval(input().strip())
#             root_node = construct_tree(n-1, tree, rule)
#             print(get_value(root_node, float("-inf"), float("inf")))
#             # print out unvisited nodes
#             print(' '.join(
#                 [str(node) for node in get_unvisited_nodes(root_node)]))
#         except EOFError:
#             break
#
#
if __name__ == '__main__':
    pp.width = pp.height = 20
    board = [[0 for i in range(pp.width)] for j in range(pp.height)]
    board[5][0] = board[7][0] = 1
    board[6][1] = board[6][2] = 2
    MiniMax(board)
    cor, rootnode = MiniMax(board)
    print(cor)
    print(rootnode.value)