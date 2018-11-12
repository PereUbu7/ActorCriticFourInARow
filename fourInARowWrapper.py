import numpy as np
from gym import spaces
import gym
import fourInARow

class ActionSpace(spaces.Discrete):
    def __init__(self, size):
        self.high = fourInARow.width
        self.low = 0

        super().__init__(size)

class FourInARowWrapper(gym.Env):

    def __init__(self, player):
        self.player = player
        self.action_space = ActionSpace(fourInARow.width)
        #self.action_space = ActionSpace([0], [8])
        fourInARow.init(player)

        self.state = self.getHotEncodedState2d()

    def ansi(self, style):
        return "\033[{0}m".format(style)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.random_sample()

    def reset(self, player):
        fourInARow.init(player)
        self.player = player
        return self.getHotEncodedState2d()

    def step(self, action):
        fourInARow.drop_disc(int(action))
        reward = 0
        if fourInARow.state != "Playing":
            if fourInARow.winner == fourInARow.player:
                reward = -1
            elif fourInARow.winner == (fourInARow.player ^ 3):
                reward = 1
            else:
                reward = 0.5
            done = True
        else:
            done = False

        stateOneHotEncoded = self.getHotEncodedState2d()

        self.state = stateOneHotEncoded

        return (stateOneHotEncoded, reward, done, None)

    def getAvaliableColumns(self):
        return np.reshape(np.array(fourInARow.get_available_cols()).astype(np.float32), (fourInARow.width))

    def render(self, mode='human'):
        if fourInARow.player == 1:
            player = "X"
        else:
            player = "O"

        print("Player:", player, "\n")
        row = "  "
        for n in range(fourInARow.width):
            row += str(n+1) + "   "
        print(row)

        row = "|"
        for _ in range(fourInARow.width):
            row += "---|"
        print(row)

        for y in range(fourInARow.height):
            row = "|"
            for x in range(fourInARow.width):
                color = 30 + fourInARow.board[x][fourInARow.height-y-1]
                character = "   "

                if fourInARow.board[x][fourInARow.height - y - 1] == 1:
                    character = " X "
                elif fourInARow.board[x][fourInARow.height - y - 1] == 2:
                    character = " O "

                if fourInARow.latest == (x, fourInARow.height-y-1):
                    color += 10
                row += self.ansi(color) + character + self.ansi(0) + "|"

            print(row)

            row = "|"
            for _ in range(fourInARow.width):
                row += "---|"
            print(row)

        #print("\n")

    def close(self):
        pass

    def getHotEncodedState(self):
        board = np.reshape(np.array(fourInARow.board), fourInARow.height * fourInARow.width)
        boardOneHotEncoded = np.zeros(fourInARow.height * fourInARow.width * 2)

        player = fourInARow.player
        playerOneHotEncoded = np.zeros(2)

        if player == 1:
            playerOneHotEncoded[0] = 1
            playerOneHotEncoded[1] = 0
        elif player == 2:
            playerOneHotEncoded[0] = 0
            playerOneHotEncoded[1] = 1

        for i in range(board.size):
            if board[i] == 1:
                boardOneHotEncoded[2 * i] = 1
                boardOneHotEncoded[2 * i + 1] = 0
            elif board[i] == 2:
                boardOneHotEncoded[2 * i] = 0
                boardOneHotEncoded[2 * i + 1] = 1
            else:
                boardOneHotEncoded[2 * i] = 0
                boardOneHotEncoded[2 * i + 1] = 0

        return np.concatenate([playerOneHotEncoded, boardOneHotEncoded])

    def getHotEncodedState2d(self):
        board = np.array(fourInARow.board)
        boardOneHotEncoded = np.resize(np.expand_dims(np.zeros(board.shape), axis=2), (7,6,2))

        player = fourInARow.player
        playerOneHotEncoded = np.zeros(2)

        if player == 1:
            playerOneHotEncoded[0] = 1
            playerOneHotEncoded[1] = 0
        elif player == 2:
            playerOneHotEncoded[0] = 0
            playerOneHotEncoded[1] = 1

        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x][y] == 1:
                    boardOneHotEncoded[x][y][0] = 1
                    boardOneHotEncoded[x][y][1] = 0
                elif board[x][y] == 2:
                    boardOneHotEncoded[x][y][0] = 0
                    boardOneHotEncoded[x][y][1] = 1
                else:
                    boardOneHotEncoded[x][y][0] = 0
                    boardOneHotEncoded[x][y][1] = 0

        return (playerOneHotEncoded, boardOneHotEncoded)

    def getCurrentPlayer(self):
        return fourInARow.player

    def renderHotEncodedState(self, hotEncodedState):
        hotEncodedPlayer = hotEncodedState[0:2:1]
        hotEncodedBoard = hotEncodedState[2::1]

        if hotEncodedPlayer[0] == 1:
            player = "X"
        elif hotEncodedPlayer[1] == 1:
            player = "O"
        else:
            print("No player in state")

        print("Player:", player, "\n")
        row = "  "
        for n in range(fourInARow.width):
            row += str(n+1) + "   "
        print(row)

        row = "|"
        for _ in range(fourInARow.width):
            row += "---|"
        print(row)

        for y in range(fourInARow.height):
            row = "|"
            for x in range(fourInARow.width):
                color = 30# + hotEncodedBoard[2*x + (fourInARow.height-2*y)*fourInARow.width-1]
                character = "   "

                if hotEncodedBoard[2*((x+1)*fourInARow.height - 1 - y)] == 1:
                    character = " X "
                elif hotEncodedBoard[2*((x+1)*fourInARow.height - 1 - y) + 1] == 1:
                    character = " O "

                row += self.ansi(color) + character + self.ansi(0) + "|"

            print(row)

            row = "|"
            for _ in range(fourInARow.width):
                row += "---|"
            print(row)

    def invertBoard(self, inBoard):
        invertedBoard = np.array(inBoard)

        board_shape = inBoard.shape

        #print("Shape:", board_shape)

        for x in range(board_shape[0]):
            for y in range(board_shape[1]):
                invertedBoard[x][y][0] = inBoard[x][y][1]
                invertedBoard[x][y][1] = inBoard[x][y][0]

        return invertedBoard
