import numpy as np
from numpy.typing import ArrayLike
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate, run_manual
from itertools import product

## valuIterationAgentv3
#
# DP 활용, ValueIteration
# row, col, dir에 대해 각각 차원 대응
class ValueIterationAgentV3(GridAdventureAgent):
    LEFT, RIGT, FWRD, PICK, DROP, ULOC = (
        GridAdventureAgent.ACTION_LEFT,
        GridAdventureAgent.ACTION_RIGHT,
        GridAdventureAgent.ACTION_FORWARD,
        GridAdventureAgent.ACTION_PICKUP,
        GridAdventureAgent.ACTION_DROP,
        GridAdventureAgent.ACTION_UNLOCK
    )
    
    ACTIONS = {
        0: LEFT, 1: RIGT, 2: FWRD, 3: PICK, 4: DROP, 5:ULOC
    }
    
    DIR_COORD = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1)   # LEFT
    }
    
    def __init__(self, desc: ArrayLike = None, n_steps: int = 1000, gamma: float = 0.9, theta: float = 1e-6, init_V: int = 0):
        '''
        desc = 초기 상태의 observation (n-dim Array | None)
        n_step = 최대 반복할 횟수
        gamma = discount factor
        theta = 학습률
        '''
        self.n_steps = n_steps
        self.gamma = gamma
        self.theta = theta
        self.init_V = init_V
        self.rows, self.cols = (26, 26)
        self.dirs = 4

        self.V = None
        self.PI = None
        
        self.desc = desc
        if self.desc is None:   ## state를 받아오지 못했을 경우.
            raise ValueError("상태 맵을 불러오지 못했습니다.")
        
    def _get_position(self, state):
        for row in range(26):
            for col in range(26):
                # print(state)
                if state[row, col].startswith('A'):
                    dir = state[row, col][1]
                    return (row, col, dir)
        return None

    def _has_key(self, state):
        for row in range(26):
            for col in range(26):
                if state[row, col] == 'KR': ## Red Key가 보드에 있다면
                    return 0
        return 1
        
    def _lock_or_unlock(self, state):
        for row in range(26):
            for col in range(26):
                if state[row, col] == 'DRL': ## Red Key가 보드에 있다면
                    return 0
        return 1
    
    def fit(self):
        V = np.zeros((self.rows, self.cols, self.dirs, 2, 2)) + self.init_V
        goals = np.argwhere(self.desc == 'G')

        # 각 목표 지점에 대해 방향에 관계없이 V를 0으로 설정
        for x,y in goals:
            V[x, y, :, :, :] = 0
    
        PI = np.zeros((self.rows, self.cols, self.dirs, 2, 2))
        
        for step in range(self.n_steps):
            delta = 0

            for row, col, dir, k, l in product(range(self.rows), range(self.cols), range(self.dirs), range(2), range(2)):
                prev_V = V[row, col, dir, k, l]
                max_V = -np.inf
                
                best_action = None
                for action in self.ACTIONS:
                    new_V = self.estimate(self.desc, V, row, col, dir, k, l, action)
                    max_V = max(max_V, new_V)
                    best_action = action
                V[row, col, dir, k, l] = max_V
                PI[row, col, dir, k, l] = best_action
                delta = max(delta, np.abs(prev_V - max_V))
                
            print("step %3d"%step)
            # 수렴했다면 종료
            if delta < self.theta:
                print("done.")
                break

        # 최적 상태 찾기
        for row, col, dir, k, l in product(range(self.rows), range(self.cols), range(self.dirs), range(2), range(2)):
            max_V = -np.inf
            max_action = -1
            for action in self.ACTIONS:
                v = self.estimate(self.desc, V, row, col, dir, k, l, action)
                if max_V < v:
                    max_V = v
                    max_action = action
            PI[row, col, dir, k, l] = max_action

        self.V = V
        self.PI = PI
        return self
        
    def estimate(self, state, V, row, col, dir, k, l, action):
        ## turn
        if action == self.LEFT:
            dir = (dir-1) % 4
        elif action == self.RIGT:
            dir = (dir+1) % 4
        
        elif action == self.FWRD:
            d_row, d_col = self.DIR_COORD[dir]
            row = np.clip(row + d_row, 0, self.rows - 1)
            col = np.clip(col + d_col, 0, self.cols - 1)
        elif action == self.PICK:
            d_row, d_col = self.DIR_COORD[dir]
            p_row = np.clip(row + d_row, 0, self.rows - 1)
            p_col = np.clip(col + d_col, 0, self.cols - 1)
            
            if state[p_row, p_col] == 'KR' and not k: 
                k = 1
                reward = 100
                return reward + self.gamma * V[row, col, dir, k, l]
        
        elif action == self.ULOC:
            d_row, d_col = self.DIR_COORD[dir]
            p_row = np.clip(row + d_row, 0, self.rows - 1)
            p_col = np.clip(col + d_col, 0, self.cols - 1)
            
            if state[p_row, p_col] == 'DRL' and k and not l: 
                l = 1
                reward = 100
                return reward + self.gamma * V[row, col, dir, k, l]
            else:
                reward = -10
                return reward + self.gamma * V[row, col, dir, k, l]
            
        reward = -1
        if state[row, col] == 'L':
            reward = -500
        if state[row, col] == 'G':
            reward = 700
        if state[row, col] == 'W':
            '''
            Goal의 위치에 가까워질수록(유클리디안 거리), Wall의 음의 보상값 증가
            '''
            g_position = np.argwhere(state == 'G')
            g_row, g_col = g_position[0]  # G의 위치
            distance_to_G = np.sqrt((row - g_row) ** 2 + (col - g_col) ** 2)
            reward = -5000 * (1 - distance_to_G / (46))
        if state[row, col] == 'DBL':
            reward = -1000
        if state[row, col] == 'DGL':
            reward = -1000
        if state[row, col] == 'DRL':
            reward = -1
        # if state[row, col] == 'DRO':
        #     reward = -1         # same as 'E'
        if state[row, col] == 'KB':
            reward = -100
        if state[row, col] == 'KG':
            reward = -100
        if state[row, col] == 'KR':
            reward = 1 if not k else -100
            
        return reward + self.gamma * V[row, col, dir, k, l]

    def act(self, state):
        row, col, dir = self._get_position(state)
        k = self._has_key(state)
        l = self._lock_or_unlock(state)
        if dir == 'U':
            idir = 0
        if dir == 'R':
            idir = 1
        if dir == 'D':
            idir = 2
        if dir == 'L':
            idir = 3
        
        action = self.PI[row, col, idir, k, l].astype(int)
        print((row,col,dir,k,l), action)
        return action

    
def train():
    env = make_grid_adventure(
        show_screen=True #False
    )
    
    state, _ = env.reset()
    agent = ValueIterationAgentV3(desc=state).fit()
    
    print(agent)
    return agent
    
if __name__ == '__main__':
    agent = train()
    evaluate(agent)
    # run_manual()