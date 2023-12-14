import numpy as np
import numpy.typing as npt
from functools import reduce

class Slime:
    def __init__(self, 
                 petri: npt.NDArray[np.float64], 
                 begin: tuple[int,int],
                 final: set[tuple[int, int]],
                 epsilon: float = .1) -> None:
        if len(petri.shape) != 2: raise ValueError('only 2D arrays for now')
        if petri[*begin] == -1: raise ValueError('cant start on a wall')
        
        self.max_x, self.max_y = petri.shape
        
        self.mold = np.zeros(petri.shape)
        self.mold[*begin] = 1
        self.can_grow = { begin } # can calculate actions [a_1, ... , a_n] from here using find_potential

        self.final = final
        self.is_end_lambda = lambda: reduce(lambda accum, next: accum and self.mold[*next], final, 1)
        
        self.epsilon = epsilon
        
        self.rewards = 0
        self.track_rewards = []
        
        # state s = { all filled cells } - that is too big of a space to search
        # approx state as (number of filled cells, the x length, and y length)
        self.state = (1, )

        # action a = cell to fill
        self.actions = { (x,y) for y in range(self.max_x) for x in range(self.max_x) }
        self.action = begin

    def is_end(self):
        return reduce(lambda accum, next: accum and self.mold[*next], self.final, 1)
    
    def find_potential(self, petri):
        potential_growth = set()
        cant_grow = []
        
        for x,y in self.can_grow:
            added = 0
            x_minus, y_minus, x_plus, y_plus = x - 1, y - 1, x + 1, y + 1
            
            # check if the spot is in bounds, there isnt mold there already
            if x_minus >= 0        and self.mold[x_minus][y] == 0 and petri[x_minus, y] >= -1: potential_growth.add((x_minus, y)); added += 1
            if y_minus >= 0        and self.mold[x][y_minus] == 0 and petri[x, y_minus] >= -1: potential_growth.add((x, y_minus)); added += 1
            if x_plus < self.max_x and self.mold[x_plus][y] == 0  and petri[x_plus, y] >= -1:  potential_growth.add((x_plus, y)); added += 1
            if y_plus < self.max_y and self.mold[x][y_plus] == 0  and petri[x, y_plus] >= -1:  potential_growth.add((x, y_plus)); added += 1
            
            if added == 0: cant_grow.append((x,y))
            
        for position in cant_grow: self.can_grow.remove(position)
        
        return potential_growth

    def prefer_max(self, petri, potential_growth):
        max_oat = 0
        for x,y in potential_growth: max_oat = petri[x, y] if petri[x, y] > max_oat else max_oat
        
        preferred_growth = np.array([np.array([x, y, petri[x, y] / max_oat]) for x,y in potential_growth], dtype=float)
        if len(preferred_growth):
            rng = np.random.default_rng()
            rng.shuffle(preferred_growth, axis=0)
            # preferred_growth[:, 2].sort()
            # preferred_growth = np.flip(preferred_growth, axis=0)

        return preferred_growth
    
    def prefer_der_max(self, petri, potential_growth):
        def find_max_der(x: int, y: int, petri):
            max_der = 0
            potential_oat = petri[x, y]
            x_minus, y_minus, x_plus, y_plus = x - 1, y - 1, x + 1, y + 1
            check_x_minus, check_y_minus, check_x_plus, check_y_plus = x_minus >= 0, y_minus >= 0, x_plus < self.max_x, y_plus < self.max_y
            
            # check if the spot is in bounds,  there isnt mold there already, there isnt a wall in the petri dish, and if the difference is bigger than the current max_der
            if check_x_minus                   and self.mold[x_minus][y] == 0       and petri[x_minus, y] >= 0       and petri[x_minus, y] - potential_oat > max_der:       max_der = petri[x_minus, y] - potential_oat
            if check_y_minus                   and self.mold[x][y_minus] == 0       and petri[x, y_minus] >= 0       and petri[x, y_minus] - potential_oat > max_der:       max_der = petri[x, y_minus] - potential_oat
            if check_x_plus                    and self.mold[x_plus][y] == 0        and petri[x_plus, y] >= 0        and petri[x_plus, y] - potential_oat > max_der:        max_der = petri[x_plus, y] - potential_oat
            if check_y_plus                    and self.mold[x][y_plus] == 0        and petri[x, y_plus] >= 0        and petri[x, y_plus] - potential_oat > max_der:        max_der = petri[x, y_plus] - potential_oat
            if check_x_minus and check_y_minus and self.mold[x_minus][y_minus] == 0 and petri[x_minus, y_minus] >= 0 and petri[x_minus, y_minus] - potential_oat > max_der: max_der = petri[x_minus, y_minus] - potential_oat
            if check_x_minus and check_y_plus  and self.mold[x_minus][y_plus] == 0  and petri[x_minus, y_plus] >= 0  and petri[x_minus, y_plus] - potential_oat > max_der:  max_der = petri[x_minus, y_plus] - potential_oat
            if check_x_plus and check_y_plus   and self.mold[x_plus][y_plus] == 0   and petri[x_plus, y_plus] >= 0   and petri[x_plus, y_plus] - potential_oat > max_der:   max_der = petri[x_plus, y_plus] - potential_oat
            if check_x_plus and check_y_minus  and self.mold[x_plus][y_minus] == 0  and petri[x_plus, y_minus] >= 0  and petri[x_plus, y_minus] - potential_oat > max_der:  max_der = petri[x_plus, y_minus] - potential_oat

            return max_der
        
        preferred_growth = np.array([np.array([x, y, find_max_der(x, y, petri)]) for x,y in potential_growth], dtype=float)
        max_der = np.max(preferred_growth[:, 2])
        preferred_growth[:, 2] /= max_der if max_der != 0 else 1

        rng = np.random.default_rng()
        rng.shuffle(preferred_growth, axis=0)

        # shuffling seems like a better way to go but can sort by preference
        # preferred_growth[:, 2].sort()
        # preferred_growth = np.flip(preferred_growth, axis=0)

        return preferred_growth
    
    def get_new_state(self, action: tuple[int, int]):
        self.mold[*action] = 1
        self.can_grow.add(action)

        self.state = (np.sum(self.mold, dtype=int), )
        return self.state


    def grow(self, petri, q):
        # not all actions are available given the state of the mold
        potential_growth = self.find_potential(petri)
        rng = np.random.default_rng()

        init_action = potential_growth.pop()
        action = init_action
        max_reward = q[(*self.state, *action)]

        for cell in potential_growth:
            next_reward = q[(*self.state, *cell)]
            if next_reward > max_reward:
                max_reward = next_reward
                action = cell
        
        if (rng.random() > self.epsilon):
            tmp_list = [init_action, *potential_growth]
            action = tmp_list[np.random.randint(0, len(tmp_list))]

        self.action = action

        return self.get_new_state(action), action
    