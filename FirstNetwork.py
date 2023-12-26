# neural network definition
import numpy
import scipy.special


class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # link weight matrices,wih and who
        # weights inside the arrays are w_i_j,where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # self.wih = (numpy.random.random(self.hnodes,self.inodes)-0.5)
        # self.who = (numpy.random.random(self.onodes,self.hnodes)-0.5)
        # 权重的范围可以在-1.0到+1.0之间。为了简单起见，我们可以将上面数组中的每个值减去0.5
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        # sigmoid 函数-常用的激活函数，也称为 logistic 函数，将输入映射到 0 到 1 之间的值。  1 / (1 + exp(-x)) xp(-x) 表示 e 的 -x 次方，e 是自然对数的底数。
        self.activation_funcation = lambda x: scipy.special.expit(x)
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targes = numpy.array(targets_list, ndmin=2).T
        print("targes",targes)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_funcation(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_ouputs = self.activation_funcation(final_inputs)
        print("final_outputs",final_ouputs)
        #final_ouputs在0到1之间,targes未经过激活函数限制,永远都会有误差.
        output_errors = targes - final_ouputs
        print("output_errors", output_errors)
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 更新链接权重的值,transpose方法转置 行列互换
        self.who += self.lr * numpy.dot((output_errors * final_ouputs * (1.0 - final_ouputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    # query the neural network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_funcation(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_ouputs = self.activation_funcation(final_inputs)
        return final_ouputs


# number of input ,hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# learning rate is 0.3
learning_rate = 0.3

# create insrance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#print(n.query([1.0, 0.5, -1.5]))
#n.train([5, 3, 1.5],[5, 3, 1.5])
n.train([1.0, 0.5, 1.5],[1.0, 0.5, 1.5])
print("query",n.query([2.0, 0.5, 1.5]))
n.train([1.0, 0.5, 1.5],[1.0, 0.5, 1.5])
print("query",n.query([2.0, 0.5, 1.5]))
n.train([2.0, 0.6, 1.6],[2.0, 0.6, 1.6])
print("query",n.query([2.0, 0.5, 1.5]))
n.train([1.9, 10.5, 8],[1.9, 10.5, 8])
print("query",n.query([2.0, 0.5, 1.5]))
n.train([20.0, 0.66, 1.98],[20.0, 0.66, 1.98])
print("query",n.query([2.0, 0.5, 1.5]))

# print("query",n.query([2.0, 0.5, 1.5]))
# numpy.random.rand(3, 3) - 0.5
