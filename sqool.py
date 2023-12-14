import numpy as np
import numpy.typing as npt
from slime import Slime
from tqdm import tqdm
import pickle
import glob
import pandas as pd

class Sqool:
    def __init__(self, 
                 actions: set[tuple[int, int]], 
                 petri: npt.NDArray[np.float64], 
                 final: set[tuple[int, int]],
                 alpha: float = .7,
                 gamma: float = 0.9) -> None:
        self.actions = actions
        self.petri_dish = petri
        self.final = final
        self.track_rewards = []
        self.track_iter = []

        # values to learn, range from [0,1]
        self.alpha = alpha
        self.gamma = gamma
        
        # init Q table
        self.q = {}
        for x,y in actions:
            for filled in range(1, petri.size):
                self.q[(filled, x, y)] = 0

    def learn(self, x: int, y: int, e: float, episodes = 10000):
        ep = 0
        slime = Slime(self.petri_dish, (x,y), self.final, epsilon=e)
        print('learning (x: {}, y: {}, epsilon: {})'.format(x,y,e))
        pbar = tqdm(total=episodes)
        
        while ep < episodes:
            reward = self.petri_dish[*slime.action]
            slime.rewards += reward
            
            if slime.is_end():
                slime = Slime(self.petri_dish, (x,y), self.final, epsilon=e)
                ep += 1
                pbar.update(1)
            else:
                old_state = slime.state
                max_q = 0
                new_state, action = slime.grow(self.petri_dish, self.q)
                
                for cell in slime.find_potential(self.petri_dish):
                    q_value = (1 - self.alpha) * self.q[(*old_state, *action)] + self.alpha * (self.petri_dish[*action] + self.gamma * self.q[(*new_state, *cell)])
                    max_q = max(max_q, q_value)
                
                self.q[(*new_state, *action)] = max_q
        
        pbar.close()

    def factorial_learning(self, xy: set[tuple[int, int]], e: tuple[float, float]):
        file_to_levels = lambda s: s.split('.')[0].split('_')[1:]
        parse_levels = lambda x, y, e: (int(x), int(y), float('.' + e))
        already_run = [parse_levels(*file_to_levels(f)) for f in glob.glob('*.pickle')]
        
        for _x, _y in xy:
            for _e in e:
                if (_x, _y, _e) not in already_run:
                    self.learn(_x, _y, _e)
                    with open(f'learn_{_x}_{_y}_{str(_e).split(".")[1]}.pickle', 'wb') as p:
                        pickle.dump(self.q, p, protocol=pickle.HIGHEST_PROTOCOL)
                    # reset q
                    for i,j in self.actions:
                        for filled in range(1, self.petri_dish.size):
                            self.q[(filled, i, j)] = 0
                else: print(f'already learned {(_x, _y, _e)}')

    def demonstration(self):
        to_csv = [] # [x, y, e, rewards, iterations]
        for file in glob.glob('*.pickle'):
            x, y, e = file.split('.')[0].split('_')[1:]
            x, y, e = int(x), int(y), float('.' + e)
            print('demonstrate (x: {}, y: {}, epsilon: {})'.format(x,y,e))
            pbar = tqdm(total=1000)

            with open(file, 'rb') as p:
                q = pickle.load(p)
                for _ in range(1000):
                    slime = Slime(self.petri_dish, (x,y), self.final, epsilon=e)
                    it = 0
                    while not slime.is_end():
                        slime.grow(self.petri_dish, q)
                        it += 1
                    to_csv.append([x, y, e, slime.rewards, it])
                    pbar.update(1)
            pbar.close()
        
        pd.DataFrame(to_csv).to_csv('stats.csv', header=['x', 'y', 'e', 'rewards', 'iterations'])
    
    # def recital(self) -> make "average" gif
        

