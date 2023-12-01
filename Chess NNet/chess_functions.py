from tensorflow.keras import models, layers
from IPython.display import clear_output
import chess, chess.engine, random, numpy

engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")


def split_dims(board):
    board3d = numpy.zeros((14, 8, 8))
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            i, j = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - i][j] = 1
        for square in board.pieces(piece, chess.BLACK):
            i, j = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - i][j] = 1
    prev = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = numpy.unravel_index(move.to_square, (8, 8))
        board3d[12][7 - i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = numpy.unravel_index(move.to_square, (8, 8))
        board3d[13][7 - i][j] = 1
    board.turn = prev
    return board3d


def generate_random_games(num_games, depth):
    x, y = list(), list()
    show = num_games // 100
    for n in range(num_games):
        if (n + 1) % show == 0:
            clear_output(wait=False)
            print(f"{round(n * 100 / num_games, 0)}% done")
        board = chess.Board()
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)
            evalu = engine.analyse(board, chess.engine.Limit(depth=depth))['score'].white().score()
            if evalu:
                x.append(split_dims(board))
                y.append(evalu)
    x, y = numpy.array(x), numpy.array(y)
    y = numpy.asarray(y / abs(y).max() / 2 + 0.5, dtype=numpy.float32)
    return x, y


def save_database(games, save_dir):
    x_train, y_train = generate_random_games(games, 0)
    numpy.save(save_dir + "/x.npy", x_train)
    numpy.save(save_dir + "/y.npy", y_train)
    print("Data Saved")


def load_database(load_dir):
    x = numpy.load(load_dir + "/x.npy")
    y = numpy.load(load_dir + "/y.npy")
    print(x.shape)
    print(y.shape)
    return x, y


def build_model_residual(res_blocks, conv_size, dense_layers=0, dense_size=8):
    board3d = layers.Input(shape=(8, 8, 14))
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(board3d)
    for i in range(res_blocks):
        previous = x
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, previous])
        x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    for _ in range(dense_layers):
        x = layers.Dense(dense_size, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)
    return models.Model(inputs=board3d, outputs=x)


def build_model_conv(conv_layers, conv_size):
    board3d = layers.Input(shape=(8, 8, 14))
    x = board3d
    for _ in range(conv_layers):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)
    return models.Model(inputs=board3d, outputs=x)


def get_ai_move(board, depth_lim, model):

    def utility(state):
        board3d = split_dims(state)
        board3d = numpy.expand_dims(board3d, 0)
        return model.predict(board3d)[0][0]

    def minimax(state, depth, alpha, beta, maximizing_player):
        if depth == 0 or state.is_game_over():
            return utility(state)

        if maximizing_player:
            max_eval = -numpy.inf
            for play in state.legal_moves:
                state.push(play)
                evalu = minimax(state, depth - 1, alpha, beta, False)
                state.pop()
                max_eval = max(max_eval, evalu)
                alpha = max(alpha, evalu)
                if beta <= alpha:
                    break
            return max_eval

        else:
            min_eval = numpy.inf
            for play in state.legal_moves:
                state.push(play)
                evalu = minimax(state, depth - 1, alpha, beta, True)
                state.pop()
                min_eval = min(min_eval, evalu)
                beta = min(beta, evalu)
                if beta <= alpha:
                    break
            return min_eval

    max_move = None
    best_eval = -numpy.inf

    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth_lim - 1, -numpy.inf, numpy.inf, board.turn)
        board.pop()
        if score > best_eval:
            best_eval = score
            max_move = move

    return max_move


def play_model(model):
    board = chess.Board()
    print(board, "\n")

    while not board.is_game_over():
        if board.turn:
            plays = list(board.legal_moves)
            print(tuple(enumerate(plays)), "\n")
            move = plays[int(input("Choose: "))]
        else:
            move = get_ai_move(board, 2, model)

        board.push(move)
        clear_output(wait=False)
        print(board)
