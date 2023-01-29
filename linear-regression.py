#Tutorial by NeuralNine

import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_csv('data.csv')

#defining functions
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error/float(len(points))

#define the gradients descent functions
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))
        b_gradient += -(2/n) * (y-(m_now * x + b_now))
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return [m, b]

#train the model, definethe initial values and apply the functions
m = 0
b = 0
L = 0.00001
epochs = 100 # the date and time relative to which a computer's clock and timestamp values are determined

for i in range(epochs):
    m, b = gradient_descent(m, b , points, L)
print(m, b)

#Animate the training process with matplotlib
#USing the ion function to turn on the interactive mode
#define two subplots in our figure. 
# The first one will visualize our data points and our regression line
# the second will plot the development of the loss function output.

plt.ion()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

#set the limits of the second plot
ax2.set_xlim([0,epochs])
ax2.set_ylim([0, loss_function(m, b, points)])

#The x-value will increase with every epoch and the y-value will show the loss function output
ax1.scatter(points.iloc[:,0], points.iloc[:,1])
line, = ax1.plot(range(20,80), range(20,80), color='red')
line2, = ax2.plot(0,0)

#define 2 empty lists
xlist = []
ylist = []

#place the loop with the training function
#with every iteration, update the graphs by using the func set_xdata, set_ydata
for i in range(epochs):
    m, b = gradient_descent(m, b, points, L)
    line.set_ydata(m * range(20,80) + b)

    xlist.append(i)
    ylist.append(loss_function(m, b, points))
    line2.set_xdata(xlist)
    line2.set_ydata(ylist)

    #update the plot in every interation using the draw function
    #to make the changs visible
    fig.canvas.draw()
  
plt.ioff() #turn off the interactive mode
plt.show() #show the plot

# Yassified version
fig.set_facecolor('#f78fda')
ax1.set_title('Linear Regression', color='white')
ax2.set_title('Loss Function', color = 'white')
ax1.grid(True, color = '#323232')
ax2.grid(True, color = '#323232')
ax1.set_facecolor('black')
ax2.set_facecolor('black')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
plt.tight_layout()

ax1.scatter(points.iloc[:,0], points.iloc[:,1], color='#EF6C35')
line, = ax1.plot(range(20, 80), range(20, 80), color='#63ccf2')
line2, = ax2.plot(0,0, color='#63ccf2')