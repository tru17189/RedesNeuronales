{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mnist_reader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import functools "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.optimize import minimize "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Sigmoid(x):\n",
    "    return 1.0/(1.0 + np.exp(-x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Sigmoid_derivada(x):\n",
    "    return sigmoide(x)*(1.0-sigmoide(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tanh(x):\n",
    "    return np.tanh(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tanh_derivada(x):\n",
    "    return 1.0 - x**2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')\n",
    "X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')\n",
    "m = len(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def inflate_matrixes(flat_thetas, shapes):\n",
    "    layers = len(shapes) + 1\n",
    "    sizes = [shape[0] * shape[1] for shape in shapes]\n",
    "    steps = np.zeros(layers, dtype=int)\n",
    "    \n",
    "    for i in range(layers-1):\n",
    "        steps[i+1] = steps[i] + sizes[i]\n",
    "        return[\n",
    "            flat_thetas[steps[i]: steps[i+1]].reshape(*shapes[i])\n",
    "            for i in range(layers-1)\n",
    "        ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def feed_forward(thetas, x):\n",
    "    a = [x]\n",
    "    for i in range(len(thetas)):\n",
    "        a.append(\n",
    "            sigmoid(\n",
    "                np.matmul(\n",
    "                    np.hstack((\n",
    "                        np.ones(len(x)).reshape(len(x), 1),\n",
    "                        a[1]\n",
    "                    )),\n",
    "                    thetas[1].T\n",
    "                )\n",
    "            )\n",
    "        )\n",
    "    return a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cost_function(flat_thetas, shapes, X, y):\n",
    "    a = feed_forward(\n",
    "        inflate_matrixes(flat_thetas, shapes),\n",
    "        x\n",
    "    )\n",
    "    return -(y * np.log(a[-1]) + (1 - y) * np.log(1 - a[-1])).sum() / len(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def backpropagation(flat_thetas, shapes, X, Y):\n",
    "    m, layers = len(x), len(shapes) + 1\n",
    "    thetas = inflate_matrixes(flat_thetas, shapes)\n",
    "    a = feed_forward(thetas, X)\n",
    "    deltas = [*range(layers -1), a[-1] - Y]\n",
    "    \n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
