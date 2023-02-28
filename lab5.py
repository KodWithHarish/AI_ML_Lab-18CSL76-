import numpy as np

x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch=7000
lr=0.1

inputlayer_neurons=2
hiddenlayers_neurons=3
output_neurons=1

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayers_neurons))
bh=np.random.uniform(size=(1,hiddenlayers_neurons))

wout=np.random.uniform(size=(hiddenlayers_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    hinpl=np.dot(x,wh)
    hinp=hinpl+bh
    hlayer_act=sigmoid(hinp)
    outinpl=np.dot(hlayer_act,wout)
    outinp=outinpl+bout
    output=sigmoid(outinp)

EO=y-output
outgrad=derivatives_sigmoid(output)
d_output=EO*outgrad

EH=d_output.dot(wout.T)
hiddengrad=derivatives_sigmoid(hlayer_act)
d_hiddenlayer=EH*hiddengrad

wout+=hlayer_act.T.dot(d_output)*lr
wh+=x.T.dot(d_hiddenlayer)*lr

print("input:\n"+str(x))
print("actual output:\n"+str(y))
print("predicted output:\n",output)