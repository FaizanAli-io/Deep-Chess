from copy import deepcopy
import pygame
import sys
import socket
import pickle
import numpy


class ChessSocket:
    def __init__(self):
        PORT = 8000
        FORMAT = 'utf-8'
        SERVER = '192.168.10.5'
        ADDR = (SERVER, PORT)

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(ADDR)

        room = "6666"  # input('Enter 4 digit room number: ')
        self.transmit(room)

    def check(self):
        while True:
            data = pickle.loads(self.client.recv(1024))
            if data == 'White' or data == 'Black':
                game.winner = data
                continue
            game.state = data
            game.turn = 1 if game.turn == -1 else -1

    def transmit(self, data):
        data = pickle.dumps(data)
        self.client.send(data)


class Chess:
    val = {
        "WP": 1,
        "WN": 3,
        "WB": 4,
        "WR": 5,
        "WQ": 9,
        "WK": 1000,
        "BP": -1,
        "BN": -3,
        "BB": -4,
        "BR": -5,
        "BQ": -9,
        "BK": -1000
    }

    def __init__(self):
        self.turn = 1
        self.selected = False

        starting_state = [
            [self.val["BR"], self.val["BN"], self.val["BB"], self.val["BQ"],
                self.val["BK"], self.val["BB"], self.val["BN"], self.val["BR"]],
            [self.val["BP"], self.val["BP"], self.val["BP"], self.val["BP"],
                self.val["BP"], self.val["BP"], self.val["BP"], self.val["BP"]],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [self.val["WP"], self.val["WP"], self.val["WP"], self.val["WP"],
                self.val["WP"], self.val["WP"], self.val["WP"], self.val["WP"]],
            [self.val["WR"], self.val["WN"], self.val["WB"], self.val["WQ"],
                self.val["WK"], self.val["WB"], self.val["WN"], self.val["WR"]]

        ]

        self.current_state = ChessState(
            starting_state, self.turn, True, True)

        self.best_move = False

    def choose_action(self, i, j):
        if self.current_state.winner:
            return

        if (self.current_state.state[i][j] > 0 and self.turn == 1) or (
                self.current_state.state[i][j] < 0 and self.turn == -1):
            self.selected = [i, j]

        elif self.selected:
            x, y = self.selected
            action = ((x, y), (i, j))

            if action in self.current_state.possible_moves:
                self.current_state = self.current_state.push_action(action)
                self.selected = False
                self.turn *= -1

                if not self.current_state.winner:
                    # if self.turn == -1:
                    #     self.best_move = choose_move(self.current_state)
                    # else:
                    #     self.best_move = False
                    self.best_move = choose_move(self.current_state)

            else:
                self.selected = False


class ChessState:
    def __init__(self, state, turn, wc, bc):
        self.turn = turn
        self.state = state
        self.white_castle = wc
        self.black_castle = bc
        self.get_color = lambda cell: 1 if cell > 0 else -1 if cell < 0 else False
        self.winner = self.get_winner()
        self.possible_moves = self.get_moves()

    def bishop(self, i, j):
        x, y = i+1, j+1
        moves = list()
        while 0 <= x < 8 and 0 <= y < 8:
            if self.state[x][y] == 0:
                moves.append(((i, j), (x, y)))
            if self.get_color(self.state[x][y]) == self.enemy:
                moves.append(((i, j), (x, y)))
                break
            if self.get_color(self.state[x][y]) == self.turn:
                break
            x += 1
            y += 1
        x, y = i+1, j-1
        while 0 <= x < 8 and 0 <= y < 8:
            if self.state[x][y] == 0:
                moves.append(((i, j), (x, y)))
            if self.get_color(self.state[x][y]) == self.enemy:
                moves.append(((i, j), (x, y)))
                break
            if self.get_color(self.state[x][y]) == self.turn:
                break
            x += 1
            y -= 1
        x, y = i-1, j+1
        while 0 <= x < 8 and 0 <= y < 8:
            if self.state[x][y] == 0:
                moves.append(((i, j), (x, y)))
            if self.get_color(self.state[x][y]) == self.enemy:
                moves.append(((i, j), (x, y)))
                break
            if self.get_color(self.state[x][y]) == self.turn:
                break
            x -= 1
            y += 1
        x, y = i-1, j-1
        while 0 <= x < 8 and 0 <= y < 8:
            if self.state[x][y] == 0:
                moves.append(((i, j), (x, y)))
            if self.get_color(self.state[x][y]) == self.enemy:
                moves.append(((i, j), (x, y)))
                break
            if self.get_color(self.state[x][y]) == self.turn:
                break
            x -= 1
            y -= 1
        return moves

    def rook(self, i, j):
        moves = list()
        for x in range(i+1, 8):
            if self.state[x][j] == 0:
                moves.append(((i, j), (x, j)))
            elif self.get_color(self.state[x][j]) == self.enemy:
                moves.append(((i, j), (x, j)))
                break
            elif self.get_color(self.state[x][j]) == self.turn:
                break
        for x in range(i-1, -1, -1):
            if self.state[x][j] == 0:
                moves.append(((i, j), (x, j)))
            elif self.get_color(self.state[x][j]) == self.enemy:
                moves.append(((i, j), (x, j)))
                break
            elif self.get_color(self.state[x][j]) == self.turn:
                break
        for y in range(j+1, 8):
            if self.state[i][y] == 0:
                moves.append(((i, j), (i, y)))
            elif self.get_color(self.state[i][y]) == self.enemy:
                moves.append(((i, j), (i, y)))
                break
            elif self.get_color(self.state[i][y]) == self.turn:
                break
        for y in range(j-1, -1, -1):
            if self.state[i][y] == 0:
                moves.append(((i, j), (i, y)))
            elif self.get_color(self.state[i][y]) == self.enemy:
                moves.append(((i, j), (i, y)))
                break
            elif self.get_color(self.state[i][y]) == self.turn:
                break
        return moves

    def king(self, i, j):
        moves = list()
        for x, y in zip([-1, -1, -1, 0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]):
            if 0 <= i + x < 8 and 0 <= j + y < 8:
                if self.get_color(self.state[i+x][j+y]) != self.turn:
                    moves.append(((i, j), (i+x, j+y)))
        if self.turn == 1 and self.white_castle:
            if self.state[7][7] == 5 and self.state[7][5] == self.state[7][6] == 0:
                moves.append(((i, j), (7, 6)))
            if self.state[7][0] == 5 and self.state[7][1] == self.state[7][2] == self.state[7][3] == 0:
                moves.append(((i, j), (7, 2)))
        elif self.turn == -1 and self.black_castle:
            if self.state[0][7] == -5 and self.state[0][5] == self.state[0][6] == 0:
                moves.append(((i, j), (0, 6)))
            if self.state[0][0] == -5 and self.state[0][1] == self.state[0][2] == self.state[0][3] == 0:
                moves.append(((i, j), (0, 2)))
        return moves

    def knight(self, i, j):
        moves = list()
        for x, y in zip([1, 1, -1, -1, 2, 2, -2, -2], [2, -2, 2, -2, 1, -1, 1, -1]):
            if 0 <= i + x < 8 and 0 <= j + y < 8:
                if self.get_color(self.state[i+x][j+y]) != self.turn:
                    moves.append(((i, j), (i+x, j+y)))
        return moves

    def black_pawn(self, i, j):
        moves = list()
        if self.state[i+1][j] == 0:
            moves.append(((i, j), (i+1, j)))
            if i == 1:
                if self.state[i+2][j] == 0:
                    moves.append(((i, j), (i+2, j)))
        if j + 1 <= 7:
            if self.get_color(self.state[i+1][j+1]) == self.enemy:
                moves.append(((i, j), (i+1, j+1)))
        if j - 1 >= 0:
            if self.get_color(self.state[i+1][j-1]) == self.enemy:
                moves.append(((i, j), (i+1, j-1)))
        return moves

    def white_pawn(self, i, j):
        moves = list()
        if self.state[i-1][j] == 0:
            moves.append(((i, j), (i-1, j)))
            if i == 6:
                if self.state[i-2][j] == 0:
                    moves.append(((i, j), (i-2, j)))
        if j + 1 <= 7:
            if self.get_color(self.state[i-1][j+1]) == self.enemy:
                moves.append(((i, j), (i-1, j+1)))
        if j - 1 >= 0:
            if self.get_color(self.state[i-1][j-1]) == self.enemy:
                moves.append(((i, j), (i-1, j-1)))
        return moves

    def get_moves(self):
        if self.winner:
            return

        moves = list()
        self.enemy = self.turn * -1

        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.turn == 1:
                    if self.state[i][j] == Chess.val["WP"]:
                        moves += self.white_pawn(i, j)
                    elif self.state[i][j] == Chess.val["WK"]:
                        moves += self.king(i, j)
                    elif self.state[i][j] == Chess.val["WN"]:
                        moves += self.knight(i, j)
                    elif self.state[i][j] == Chess.val["WB"]:
                        moves += self.bishop(i, j)
                    elif self.state[i][j] == Chess.val["WR"]:
                        moves += self.rook(i, j)
                    elif self.state[i][j] == Chess.val["WQ"]:
                        moves += self.rook(i, j)
                        moves += self.bishop(i, j)

                elif self.turn == -1:
                    if self.state[i][j] == Chess.val["BP"]:
                        moves += self.black_pawn(i, j)
                    elif self.state[i][j] == Chess.val["BK"]:
                        moves += self.king(i, j)
                    elif self.state[i][j] == Chess.val["BN"]:
                        moves += self.knight(i, j)
                    elif self.state[i][j] == Chess.val["BB"]:
                        moves += self.bishop(i, j)
                    elif self.state[i][j] == Chess.val["BR"]:
                        moves += self.rook(i, j)
                    elif self.state[i][j] == Chess.val["BQ"]:
                        moves += self.rook(i, j)
                        moves += self.bishop(i, j)

        return moves

    def push_action(self, action):
        if self.winner:
            return

        newstate = deepcopy(self)
        (x, y), (i, j) = action

        # castling check
        if newstate.state[x][y] == 1000:
            newstate.white_castle = False
            if j - y == 2:
                newstate.state[7][5] = 5
                newstate.state[7][7] = 0
            elif j - y == -2:
                newstate.state[7][3] = 5
                newstate.state[7][0] = 0

        elif newstate.state[x][y] == -1000:
            newstate.black_castle = False
            if j - y == 2:
                newstate.state[0][5] = -5
                newstate.state[0][7] = 0
            elif j - y == -2:
                newstate.state[0][3] = -5
                newstate.state[0][0] = 0

        # queen promotion
        elif newstate.state[x][y] == 1 and i == 0:
            newstate.state[x][y] = 9

        elif newstate.state[x][y] == -1 and i == 7:
            newstate.state[x][y] = -9

        newstate.state[i][j] = newstate.state[x][y]
        newstate.state[x][y] = 0

        newstate.turn *= -1
        newstate.winner = newstate.get_winner()
        newstate.possible_moves = newstate.get_moves()
        return newstate

    def get_utility(self):
        if self.winner:
            return self.winner * numpy.inf
        return sum([sum(row) for row in self.state])

    def get_winner(self):
        black_win = True
        white_win = True
        for row in self.state:
            if Chess.val["WK"] in row:
                black_win = False
            if Chess.val["BK"] in row:
                white_win = False
        return 1 if white_win else -1 if black_win else 0

    def show_state(self):
        for row in self.state:
            for cell in row:
                print(str(cell).rjust(6), end=" ")
            print(end="\n")


class Renderer:
    class Box:
        def __init__(self, i, j, buffX, buffY, blocksize, surf, space=0.1):
            self.i = i
            self.j = j
            self.surf = surf
            self.position = pygame.Rect(buffX+(self.j*(blocksize+(blocksize*space))), buffY+(
                self.i*(blocksize+(blocksize*space))), blocksize, blocksize)

        def show(self, col):
            pygame.draw.rect(self.surf, col, self.position)

    def __init__(self, game):
        pygame.init()
        scW, scH = 800, 600
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((scW, scH))
        pygame.display.set_caption("Chess")
        pygame.mouse.set_cursor(*pygame.cursors.diamond)

        self.game = game
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.light_greeen = (0, 120, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

        self.font = pygame.font.SysFont('inkfree', 20, 3)
        piece_font = pygame.font.SysFont('inkfree', 50, 3)

        self.block = 60
        buffX, buffY = int((scW/2)-(((self.block+(self.block*0.1))*8)/2)
                           ), 20+int((scH/2)-(((self.block+(self.block*0.1))*8)/2))
        self.board = [[self.Box(i, j, buffX, buffY, self.block, self.screen)
                       for j in range(8)] for i in range(8)]
        self.back_button = pygame.Rect(50, 50, 50, 50)

        self.pieces = {
            Chess.val["WP"]: piece_font.render("P", True, self.blue),
            Chess.val["WN"]: piece_font.render("N", True, self.blue),
            Chess.val["WB"]: piece_font.render("B", True, self.blue),
            Chess.val["WR"]: piece_font.render("R", True, self.blue),
            Chess.val["WQ"]: piece_font.render("Q", True, self.blue),
            Chess.val["WK"]: piece_font.render("K", True, self.blue),
            Chess.val["BP"]: piece_font.render("P", True, self.black),
            Chess.val["BN"]: piece_font.render("N", True, self.black),
            Chess.val["BB"]: piece_font.render("B", True, self.black),
            Chess.val["BR"]: piece_font.render("R", True, self.black),
            Chess.val["BQ"]: piece_font.render("Q", True, self.black),
            Chess.val["BK"]: piece_font.render("K", True, self.black)
        }

    def show(self):
        self.screen.fill(self.black)

        winner = self.game.current_state.winner
        winner = "White" if winner == 1 else "Black" if winner == -1 else ""
        status_text = f"{winner} has won the game" if winner else (
            'White to move' if self.game.turn == 1 else 'Black to move')
        self.screen.blit(self.font.render(
            status_text, True, self.white), (135, 15))
        self.screen.blit(self.font.render(
            f"Current Eval: {self.game.current_state.get_utility()}", True, self.white), (555, 15))

        # pygame.draw.rect(self.screen, self.blue, self.back_button)

        [[box.show(self.white) for box in line] for line in self.board]

        if self.game.selected:
            for move in self.game.current_state.possible_moves:
                if tuple(self.game.selected) == move[0]:
                    pygame.draw.rect(
                        self.screen, self.blue, self.board[move[1][0]][move[1][1]].position, 2)

        buff = self.block * 0.1
        for i in range(8):
            for j in range(8):
                if self.game.current_state.state[i][j]:
                    self.screen.blit(self.pieces[self.game.current_state.state[i][j]], (self.board[i]
                                                                                        [j].position.x+buff, self.board[i][j].position.y+buff))
        if self.game.selected:
            pygame.draw.rect(
                self.screen, self.red, self.board[self.game.selected[0]][self.game.selected[1]].position, 2)

        if self.game.best_move:
            pygame.draw.rect(
                self.screen, self.light_greeen, self.board[self.game.best_move[0][0]][self.game.best_move[0][1]].position, 4)

            pygame.draw.rect(
                self.screen, self.green, self.board[self.game.best_move[1][0]][self.game.best_move[1][1]].position, 4)

        pygame.display.flip()
        self.clock.tick(10)

    def clicked(self, pos):
        if self.back_button.collidepoint(pos):
            pass  # self.game.current_state.pop_action()

        for line in self.board:
            for box in line:
                if box.position.collidepoint(pos):
                    self.game.choose_action(box.i, box.j)


def minimax(state, depth, alpha, beta):
    if depth == 0 or state.winner:
        return state.get_utility()

    if state.turn > 0:
        best_eval = -numpy.inf
        for play in state.possible_moves:
            newstate = state.push_action(play)
            evaluation = minimax(newstate, depth - 1, alpha, beta)
            best_eval = max(best_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break

    else:
        best_eval = numpy.inf
        for play in state.possible_moves:
            newstate = state.push_action(play)
            evaluation = minimax(newstate, depth - 1, alpha, beta)
            best_eval = min(best_eval, evaluation)
            beta = min(alpha, evaluation)
            if beta <= alpha:
                break

    return best_eval


def choose_move(state, depth=4):
    print(f"Calculating Best Move at depth = {depth}")

    mystate = deepcopy(state)
    myturn = mystate.turn
    best_score = numpy.inf * myturn * -1

    for move in mystate.possible_moves:
        newstate = mystate.push_action(move)
        score = minimax(newstate, depth - 1, -numpy.inf, numpy.inf)

        if myturn > 0:
            if score > best_score:
                best_move = move
                best_score = score

        else:
            if score < best_score:
                best_move = move
                best_score = score

    print("Best Move Calculated.\n")

    return best_move


game = Chess()
renderer = Renderer(game)

while True:
    for event in pygame.event.get():
        if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            renderer.clicked(pygame.mouse.get_pos())

    renderer.show()
