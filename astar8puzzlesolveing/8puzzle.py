
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pyautogui
import copy
import time

class Node:
    def __init__(self, data, hval, level):
        self.data = data
        self.hval = hval
        self.level = level

def movePuzzle(puzzle, x, y, oper):
    if(oper == 'up'):
        if(x - 1 < 0):
            return None
        else:
            tmp = puzzle[x][y]
            puzzle[x][y] = puzzle[x-1][y]
            puzzle[x-1][y] = tmp

            return puzzle

    elif(oper == 'down'):
        if (x + 1 >= 3):
            return None
        else:
            tmp = puzzle[x][y]
            puzzle[x][y] = puzzle[x + 1][y]
            puzzle[x + 1][y] = tmp

            return puzzle

    elif(oper == 'right'):
        if (y + 1 >= 3):
            return None
        else:
            tmp = puzzle[x][y]
            puzzle[x][y] = puzzle[x][y + 1]
            puzzle[x][y + 1] = tmp

            return puzzle

    elif(oper == 'left'):
        if (y - 1 < 0):
            return None
        else:
            tmp = puzzle[x][y]
            puzzle[x][y] = puzzle[x][y - 1]
            puzzle[x][y - 1] = tmp

            return puzzle

def checkZero(puzzle):
    x, y = 0, 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 'puzzle/num_0.png':
                x, y = i, j
    return x, y


imgList11 = ['puzzle/num_1.png', 'puzzle/num_2.png', 'puzzle/num_3.png']
imgList12 = ['puzzle/num_4.png', 'puzzle/num_5.png', 'puzzle/num_6.png']
imgList13 = ['puzzle/num_7.png', 'puzzle/num_8.png', 'puzzle/num_0.png']
imgList1 = imgList11 + imgList12 + imgList13

imgList2 = np.reshape(imgList1,(3,3))

for i in range(60):
    x, y = checkZero(imgList2)
    random_number = random.randrange(4)
    if(random_number == 0):
        movePuzzle((imgList2), x, y, 'left')
    if(random_number == 1):
        movePuzzle((imgList2), x, y, 'right')
    if(random_number == 2):
        movePuzzle((imgList2), x, y, 'up')
    if(random_number == 3):
        movePuzzle((imgList2), x, y, 'down')



fig, axes = plt.subplots(3, 3, figsize=(5, 5))


button_ax = plt.axes([0.8, 0.01, 0.1, 0.03])    #버튼 추가
button = Button(button_ax, 'solve')



img1 = cv2.imread(imgList2[0][0])
img2 = cv2.imread(imgList2[0][1])
img3 = cv2.imread(imgList2[0][2])
img4 = cv2.imread(imgList2[1][0])
img5 = cv2.imread(imgList2[1][1])
img6 = cv2.imread(imgList2[1][2])
img7 = cv2.imread(imgList2[2][0])
img8 = cv2.imread(imgList2[2][1])
img9 = cv2.imread(imgList2[2][2])
img0 = cv2.imread('puzzle/num_0.png')

plt.subplot(3, 3, 1)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img1)

plt.subplot(3, 3, 2)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img2)

plt.subplot(3, 3, 3)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img3)

plt.subplot(3, 3, 4)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img4)

plt.subplot(3, 3, 5)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img5)

plt.subplot(3, 3, 6)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img6)

plt.subplot(3, 3, 7)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img7)

plt.subplot(3, 3, 8)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img8)

plt.subplot(3, 3, 9)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.imshow(img9)



class Puzzle:
    def __init__(self, start):
        self.start = start

    def h(self, puzzle, goal):
        cnt = 0
        for i in range(3):
            for j in range(3):
                if puzzle[i][j] != goal[i][j]:
                    cnt += 1

        return cnt

    def f(self, puzzle, goal):
        return puzzle.level + self.h(puzzle.data, goal)

    def astar(self, puzzle):
        visitcheck = False
        visit = []
        queue = []
        goal = np.reshape(imgList1,(3,3))
        oper = ['up', 'down', 'right', 'left']
        start = Node(data=puzzle, hval=self.h(puzzle, goal), level=0)
        queue.append(start)
        global imgList2
        while queue:

            current = queue.pop(0)
            
            tmp = current.data
            imgList2 = current.data
            
            showSingleimg(1,2)
            showSingleimg(3,4)
            showSingleimg(5,6)
            showSingleimg(7,8)
            showSingleimg(9,1)
            plt.draw()
            plt.pause(0.1)
            #time.sleep(0.2)

            
            print("////////")
            if(self.h(current.data, goal)==0):
                print("ftest")
                return visit
            else:
                visit.append(current.data)
                x, y = checkZero(current.data)

                for op in oper:
                    visitcheck = False

                    imgList2 = current.data
           
                    showSingleimg(1,2)
                    showSingleimg(3,4)
                    showSingleimg(5,6)
                    showSingleimg(7,8)
                    showSingleimg(9,1)
                    plt.draw()
                    plt.pause(0.1)
                    #time.sleep(0.2)

                    next = movePuzzle(copy.deepcopy(current.data), x, y, op)
                    if next is not None:
                        
                        print(self.h(next, goal))
                        imgList2 = next
            
                        showSingleimg(1,2)
                        showSingleimg(3,4)
                        showSingleimg(5,6)
                        showSingleimg(7,8)
                        showSingleimg(9,1)
                        plt.draw()
                        plt.pause(0.1)
                        #time.sleep(0.2)
                    
                    
           
                        
                    
                    for node in visit:
                        
                        if np.array_equal(next, node.data):
                            visitcheck = True
                            
                    if visitcheck != True and next is not None:
                        queue.append(Node(next, self.h(next, goal), current.level + 1))
                        
                queue.sort(key=lambda x:self.f(x,goal), reverse=False)
        return -1

 

for i in range(3):
    for j in range(3):
        print(imgList2[i][j])
    print()
def button_clicked(event):
    puzzle_ = Puzzle(imgList2)
    puzzle_.astar(imgList2)
    print('Button test')

button.on_clicked(button_clicked)






def showSingleimg(a,b):
    plt.subplot(3, 3, a)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(cv2.imread(imgList2[int((a-1)/3)][(a-1)%3]))
    plt.subplot(3, 3, b)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(cv2.imread(imgList2[int((b-1)/3)][(b-1)%3]))
    
    
def add_point(event):
    #if event.inaxes != ax:
    #    return\
    global imgList2             # 외부 변수를 참조하므로  global 추가
    if event.button ==1:
        fore = pyautogui.getActiveWindow()
        pos = pyautogui.position()
        x = pos.x - fore.left
        y = pos.y - fore.top
        print("클릭 : ", x, ", ", y)

        if (x >= 70 and x <= 195) and (y >= 95 and y <= 215):
            if(imgList2[0][1] == 'puzzle/num_0.png'): 
                print("test")
                movePuzzle(imgList2,0,0,'right')
               
                showSingleimg(1,2)

                
            elif(imgList2[1][0] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,0,0,'down')
                print("test")
                showSingleimg(1,4)
            

        if (x >= 200 and x <= 325) and (y >= 100 and y <= 210):
            if (imgList2[0][0] == 'puzzle/num_0.png'):
                print("test")
                movePuzzle(imgList2,0,1,'left')
                
                showSingleimg(1,2) 

            elif (imgList2[0][2] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,0,1,'right')
                print("test")
                showSingleimg(2,3)
            elif (imgList2[1][1] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,0,1,'down')
                print("test")
                showSingleimg(2,5)

        if (x >= 330 and x <= 455) and (y >= 95 and y <= 215):
            if(imgList2[0][1] == 'puzzle/num_0.png'): # opencv  에서 이미지는 배열로 처리되기에 () 이 필요
               
                movePuzzle(imgList2,0,2,'left')
                print("test")
                showSingleimg(2,3)

                
            elif(imgList2[1][2] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,0,2,'down')
                print("test")
                showSingleimg(3,6)

        if (x >= 75 and x <= 195) and (y >= 225 and y <= 340):
            if (imgList2[0][0] == 'puzzle/num_0.png'):
               
                movePuzzle(imgList2,1,0,'down')
                print("test")
                showSingleimg(1,4)

            elif (imgList2[1][1] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,1,0,'right')
                print("test")
                showSingleimg(5,4)
            elif (imgList2[2][0] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,1,0,'down')
                print("test")
                showSingleimg(4,7)


        if (x >= 205 and x <= 325) and (y >= 225 and y <= 340):
            if (imgList2[0][1] == 'puzzle/num_0.png'):
               
                movePuzzle(imgList2,1,1,'up')
                print("test")
                showSingleimg(2,5)

            elif (imgList2[1][0] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,1,1,'left')
                print("test")
                showSingleimg(4,5)
            elif (imgList2[1][2] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,1,1,'right')
                print("test")
                showSingleimg(5,6)
            elif (imgList2[2][1] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,1,1,'down')
                print("test")
                showSingleimg(5,8)

        if (x >= 335 and x <= 455) and (y >= 225 and y <= 340):
            if (imgList2[0][2] == 'puzzle/num_0.png'):
               
                movePuzzle(imgList2,1,2,'up')
                print("test")
                showSingleimg(3,6)

            elif (imgList2[1][1] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,1,2,'left')
                print("test")
                showSingleimg(5,6)
            elif (imgList2[2][2] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,1,2,'down')
                print("test")
                showSingleimg(6,9)

        if (x >= 75 and x <= 195) and (y >= 355 and y <= 470):
            if(imgList2[1][0] == 'puzzle/num_0.png'): 
                print("test")
                movePuzzle(imgList2,2,0,'up')
               
                showSingleimg(4,7)

                
            elif(imgList2[2][1] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,2,0,'right')
                print("test")
                showSingleimg(7,8)
        if (x >= 205 and x <= 325) and (y >= 355 and y <= 470):
            if (imgList2[2][0] == 'puzzle/num_0.png'):
               
                movePuzzle(imgList2,2,1,'left')
                print("test")
                showSingleimg(7,8)

            elif (imgList2[1][1] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,2,1,'up')
                print("test")
                showSingleimg(5,8)
            elif (imgList2[2][2] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,2,1,'right')
                print("test")
                showSingleimg(8,9)


        if (x >= 335 and x <= 455) and (y >= 355 and y <= 470):
            if(imgList2[2][1] == 'puzzle/num_0.png'): 
                print("test")
                movePuzzle(imgList2,2,2,'left')
               
                showSingleimg(8,9)

                
            elif(imgList2[1][2] == 'puzzle/num_0.png'):
                movePuzzle(imgList2,2,2,'up')
                print("test")
                showSingleimg(6,9)
        plt.show()

cid = plt.connect('button_press_event', add_point)
plt.subplots_adjust(wspace=0.01, hspace=0.02)
plt.show()

