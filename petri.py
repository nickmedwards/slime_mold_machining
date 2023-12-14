import numpy as np

class Petri:
    final_range = .3

    def __init__(self, wall_locs: list[list[int]]) -> None:
        self.dish = np.zeros((len(wall_locs), len(wall_locs[0])), dtype=float)
        
        for i in range(len(wall_locs)):
            for j in range(len(wall_locs[i])):
                if wall_locs[i][j]: self.dish[i][j] = -1

        self.final = set()
    
    def add_oat(self, x: int, y: int, r: float):
        eucl = lambda _x, _y: np.sqrt((_x - x)**2 + (_y - y)**2)
        d = np.inf
        
        # find closest wall
        for i in range(self.dish.shape[0]):
            for j in range(self.dish.shape[1]):
                if self.dish[i][j] < 0:
                    e = eucl(i, j)
                    if e < d: d = e
        
        if d > r * (1 - self.final_range) and d < r * (1 + self.final_range): 
            self.final.add((x,y))

        return np.exp(1 - d / r) if d >= r else (d/r)**6
                    
    def add_oats(self, r: float):
        oats = np.zeros(self.dish.shape)
        for i in range(self.dish.shape[0]):
            for j in range(self.dish.shape[1]):
                if self.dish[i][j] == 0:
                    oats[i][j] = self.add_oat(i, j, r)
        
        # normalize oats to there are sum = 0
        oats -= np.sum(oats) / self.dish.size
        # add oats to dish
        self.dish += oats
        self.total_oats = np.sum(oats)
