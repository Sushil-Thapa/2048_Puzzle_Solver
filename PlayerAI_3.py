import multiprocessing
from random import randint
from BaseAI_3 import BaseAI
import time
import math
import numpy as np

class PlayerAI(BaseAI):

    def __init__(self):
        self.max_depth = 1
        self.state_time = time.clock()
        self.max_time = 0.2
        self.move = None

    def getMove(self,grid):
        self.start_time = time.clock()
        self.prev_grid = grid
        manager = multiprocessing.Manager()
        ret = manager.Value('move',-1)
        def decision(ret,name=''):
            prev_utility = 0
            for self.max_depth in range(0,100):
                state, move, utility = self.maximize(grid,-float("Inf"),float("Inf"),0)
                if prev_utility <= utility:
                    ret.value = move
                prev_utility = utility
                #print("max_depth %d and utility %d"%(self.max_depth,utility))
                #print(move)
        p = multiprocessing.Process(target=decision,args=(ret,'move'))
        p.start()
        p.join(0.2)
        if p.is_alive():
            p.terminate()
            p.join()
        return ret.value

    def terminal_test(self,state):
        return not state.canMove()

    def eval(self,state):
        return state.getMaxTile()

    def mul(self,pos,dom):
        return abs(dom[0]-pos[0])*abs(dom[1]-pos[1])

    def l1(self, point1, point2):
        diff= [point1[0] - point2[0], point1[1] - point2[1]]
        return abs(diff[0])+abs(diff[1])

    def dominatingCorner(self,pos):
        corner = [(0,0),(0,3),(3,3),(3,0)]
        cornerdis = [(self.l1((x,y),pos),(x,y)) for x,y in corner]
        sorted(cornerdis)
        return cornerdis[-1][-1]

    def extremeCorner(self,pos):
        if pos == (0,0) or pos == (3,3) or pos == (0,3) or pos == (3,0):
            return True

    def corner(sel,pos):
        if pos[0] == 0 or pos[0] == 3 or pos[1] == 0 or pos[1] == 3:
            return True

    def getKey(self,item):
        val, pos = item
        return val + max(5-self.l1(pos,(0,0)),5-self.l1(pos,(4,0)),5-self.l1(pos,(4,4)),5-self.l1(pos,(0,4)))

    def heuristicNp(self,state):
        board = np.array(state.map,dtype=np.int32)
        x, y = np.ogrid[0:board.shape[0],0:board.shape[1]]
        maxp= np.argmax(board)
        maxp = (maxp//board.shape[0], maxp%board.shape[1])
        maxv = board[maxp[0],maxp[1]]
        domp = self.dominatingCorner(maxp)
        hamDis = abs(domp[0]-x) + abs(domp[1]-y)
        return np.mean(2**hamDis.transpose() * board) + maxv


    def heuristic(self,state):
        return self.heuristicNp(state)
        cornerStrategy = 0
        closeToMax = 0
        mergeNeighbour = 0
        boundaryValue = 0
        noOfFreeCell = len(state.getAvailableCells())

        # list of tiles
        tiles  = [(state.map[x][y],(x,y)) for x in range(4) for y in range(4)]
        sorted(tiles,key=self.getKey,reverse=True)

        maxtile, maxp = tiles[0]

        domCorner = self.dominatingCorner(maxp)
        # Corner Stratey
        #if domCorner == maxp:
        pror = 1
        proc = 1
        if maxp == domCorner:
            for x in range(0,3):
                try:
                    pror *= math.log2(state.map[domCorner[0]][x])
                    proc *= math.log2(state.map[x][domCorner[1]])
                except ValueError:
                    pror = 0
                    proc = 0
            cornerStrategy = max(pror,proc)*max(maxtile,100)
            cornerStrategy += 36 * maxtile

        if self.corner(maxp):
            cornerStrategy += maxtile
        #else:
            #cornerStrategy -= self.l1(maxp,domCorner)* maxtile
        # proximity to maxtile
        hammdis = [0,1,1,2,2,2,3,3,3,3,4,4,4,5,5,6]

        for tile in range(1,len(tiles)):
            val, pos = tiles[tile]
            if val == 0:
                break
            """
            if maxtile == val and tile != 0:
                #choosing max point as the cornerest point
                cornerp1 = min([self.l1(pos,(0,0)),self.l1(pos,(0,4)),self.l1(pos,(4,0)),self.l1(pos,(4,4))])
                cornerp = min([self.l1(maxp,(0,0)),self.l1(maxp,(0,4)),self.l1(maxp,(4,0)),self.l1(maxp,(4,4))])
                if cornerp1 < cornerp:
                    maxtile, maxp, val, pos = val, pos, maxtile, maxp

                if self.l1(pos,maxp) <= hammdis[tile]:
                    mergeNeighbour += tile
            """

            #closeToMax += hammdis[-tile] * val
            #merge probability

            pDiff = 0
            nDiff = 0
            for posChan in [(1,0),(-1,0),(0,1),(0,-1)]:
                logVal = math.log2(val)
                try:
                    valNeigh = state.map[pos[0]+posChan[0]][pos[1]+posChan[1]]
                    if valNeigh == 0:
                        if val == 2 or val == 4:
                            boundaryValue += logVal**(16-noOfFreeCell)
                    else:
                        logNeigh = math.log2(valNeigh)
                        diff = logVal-logNeigh
                        if diff >= 0:
                            pDiff += 1
                        else:
                            nDiff += 1
                        mergeNeighbour += logVal*abs(logVal-abs(diff))*noOfFreeCell
                except IndexError:
                    pass
            if not (pDiff == 0 or nDiff == 0):
                #closeToMax += (6-self.l1(domCorner,pos)) * math.log2(val)
                closeToMax += (16-self.l1(domCorner,pos)) * math.log2(val)
                if (self.l1(domCorner,pos)<=hammdis[-tile-1]):
                    closeToMax *= hammdis[-tile-1]

        score =  (16-noOfFreeCell)/(noOfFreeCell+1) * maxtile + 1.5 * cornerStrategy + 2 * closeToMax + 2.5 * mergeNeighbour + 3 * boundaryValue
        #if score >= 2048:
        #return 1024
        #else:
        #    return score
        #score = (weight + (1-optimistic) * (1-weight)) * maxValue/2048 + (1 - weight + (1-optimistic) * (1-weight)) * extra
        return score

    def children(self,state,moves):
        childs = []
        for move in moves:
            newstate = state.clone()
            newstate.move(move)
            childs.append((newstate,move))
        return childs

    def maximize(self,state,alpha,beta,depth):
        if self.terminal_test(state):
            return None, None, self.heuristic(state)
        elif depth == self.max_depth:
            return None, None, self.heuristic(state)

        maxChild = None
        maxUtility = -float('Inf')
        rightMove = None

        self.prev_grid = state
        moves = state.getAvailableMoves()
        for (child, move) in self.children(state,moves):
            _ , gotMove , utility = self.minimize(child,alpha,beta,depth+1)
            if utility > maxUtility:
                maxChild, maxUtility, rightMove = child, utility, move
            if maxUtility > beta:
                break
            if maxUtility < alpha:
                alpha = maxUtility
        return (maxChild, rightMove, maxUtility)


    def minChildrens(self,state,cells):
        childs = []
        for cell in cells:
            for value in [2,4]:
                newstate = state.clone()
                newstate.setCellValue(cell,value)
                childs.append((newstate,(cell,value)))
        return childs

    def minimize(self,state,alpha,beta,depth):
        if self.terminal_test(state):
            return None, None, self.heuristic(state)
        elif depth == self.max_depth:
            return None, None, self.heuristic(state)

        self.prev_grid = state
        minChild, minUtility = None, float('Inf')
        cells = state.getAvailableCells()
        rightMove = None
        for (child, move) in self.minChildrens(state,cells):
            _, gotMove , utility = self.maximize(child,alpha,beta,depth+1)
            if utility < minUtility:
                maxChild, minUtility, rightMove = child, utility, move
            if minUtility <= alpha:
                break
            if minUtility < beta:
                beta = minUtility
        return (minChild,rightMove, minUtility)
