'''
This code contains all of the agents (from simple to complex- Random, Greedy, Heuristic and Minimax).
The user can play against all AIs as player 1.
It allows the user to play against the minimax AI (as it is the main focus) as player 1 or player 2.
It can also spectate the 3 AIs (Random AI excluded) to play each other.
For minimax they can alter the depth and number of random moves (if player 1).
They can uncomment 'time.sleep(1)' to watch the game move by move if the AIs are at a low depth.
'''

import tkinter as tk
from tkinter import messagebox
import random
import math
import time

# Constants for the game
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
ROWS = 6
COLS = 7

# Utility class for handling board operations
class BoardUtils:
    # Check if a move is valid
    @staticmethod
    def is_valid_move(board, col):
        return board[0][col] == EMPTY

    # Find the next open row in a column
    @staticmethod
    def get_next_open_row(board, col):
        for r in range(ROWS - 1, -1, -1):
            if board[r][col] == EMPTY:
                return r
        return -1

    # Check if there's a winner on the board
    @staticmethod
    def check_winner(board):
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] != EMPTY:
                    # Check horizontally
                    if c + 3 < COLS and board[r][c] == board[r][c + 1] == board[r][c + 2] == board[r][c + 3]:
                        return board[r][c]
                    # Check vertically
                    if r + 3 < ROWS and board[r][c] == board[r + 1][c] == board[r + 2][c] == board[r + 3][c]:
                        return board[r][c]
                    # Check diagonals
                    if c + 3 < COLS and r + 3 < ROWS and board[r][c] == board[r + 1][c + 1] == board[r + 2][c + 2] == board[r + 3][c + 3]:
                        return board[r][c]
                    if c + 3 < COLS and r - 3 >= 0 and board[r][c] == board[r - 1][c + 1] == board[r - 2][c + 2] == board[r - 3][c + 3]:
                        return board[r][c]
        return None

# Class for handling AI players
class AI:
    def __init__(self, symbol):
        self.symbol = symbol

    # Placeholder for getting a move
    def get_move(self, board):
        pass

# Class representing the game state
class GameState(BoardUtils):
    def __init__(self, board, last_move=None):
        self.board = board
        self.last_move = last_move

    # Get valid moves for the current state
    def get_valid_moves(self):
        return [col for col in range(COLS) if self.is_valid_move(self.board, col)]

    # Get the next state after a move
    def get_next_state(self, move):
        next_board = [row[:] for row in self.board]
        for row in range(ROWS - 1, -1, -1):
            if next_board[row][move] == EMPTY:
                next_board[row][move] = AI_AGENT_SYMBOL
                break
        return GameState(next_board, last_move=move)

    # Get a random next state
    def get_random_next_state(self):
        move = random.choice(self.get_valid_moves())
        return self.get_next_state(move)

    # Check if the game is terminal
    def is_terminal(self):
        return self.get_winner() is not None or all(self.board[0][col] != EMPTY for col in range(COLS))

    # Get the winner of the game
    def get_winner(self):
        return self.check_winner(self.board)

    # Get possible next states
    @staticmethod
    def get_next_states(board, symbol):
        next_states = []
        for move in range(COLS):
            if BoardUtils.is_valid_move(board, move):
                row = BoardUtils.get_next_open_row(board, move)
                next_board = [row[:] for row in board]
                next_board[row][move] = symbol
                next_states.append(next_board)
        return next_states

# Random AI class
class RandomAI(AI):
    def __init__(self, symbol):
        super().__init__(symbol)

    # Get a random move
    def get_move(self, board):
        valid_moves = [col for col in range(COLS) if BoardUtils.is_valid_move(board, col)]
        return random.choice(valid_moves)

# Greedy AI class -  similar to random but play winning moves and stop opponent's winning moves
class GreedyAI(AI):
    def __init__(self, symbol):
        super().__init__(symbol)

    # Get the next move based on a greedy strategy
    def get_move(self, board):
        for move in range(COLS):
            if BoardUtils.is_valid_move(board, move):
                row = BoardUtils.get_next_open_row(board, move)
                # Check if the move results in a win for the AI
                board[row][move] = self.symbol
                if BoardUtils.check_winner(board) == self.symbol:
                    board[row][move] = EMPTY
                    return move
                board[row][move] = EMPTY
                
                # Check if the move blocks the opponent's win
                opponent_symbol = PLAYER1 if self.symbol == PLAYER2 else PLAYER2
                board[row][move] = opponent_symbol
                if BoardUtils.check_winner(board) == opponent_symbol:
                    board[row][move] = EMPTY
                    return move
                board[row][move] = EMPTY
                
        # If no immediate win or block, make a random move
        return random.choice([col for col in range(COLS) if BoardUtils.is_valid_move(board, col)])

# Heuristic AI class- prioritises centre or column, row, diagonal with most consecutive pieces
class HeuristicAI(AI):
    def __init__(self, symbol):
        super().__init__(symbol)

    # Get the next move using a heuristic strategy
    def get_move(self, board):
        for move in range(COLS):
            if BoardUtils.is_valid_move(board, move):
                row = BoardUtils.get_next_open_row(board, move)
                # Check if the move results in a win for the AI
                board[row][move] = self.symbol
                if BoardUtils.check_winner(board) == self.symbol:
                    board[row][move] = EMPTY
                    return move
                board[row][move] = EMPTY

                # Check if the move blocks the opponent's win
                opponent_symbol = PLAYER1 if self.symbol == PLAYER2 else PLAYER2
                board[row][move] = opponent_symbol
                if BoardUtils.check_winner(board) == opponent_symbol:
                    board[row][move] = EMPTY
                    return move
                board[row][move] = EMPTY

        # If no immediate win or block, use a heuristic strategy
        return self.heuristic_move(board)

    # Heuristic move strategy
    def heuristic_move(self, board):
        # Prioritize the center column
        if BoardUtils.is_valid_move(board, COLS // 2):
            return COLS // 2

        # Otherwise, choose the position with the highest score
        max_score = float('-inf')
        best_move = None
        for move in range(COLS):
            if BoardUtils.is_valid_move(board, move):
                row = BoardUtils.get_next_open_row(board, move)
                score = self.evaluate_position(board, row, move)
                if score > max_score:
                    max_score = score
                    best_move = move

        return best_move

    # Evaluate a position on the board based on rows, columns, and diagonals
    def evaluate_position(self, board, row, col):
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            # Count consecutive pieces in the current direction
            r, c = row, col
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == self.symbol:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == self.symbol:
                count += 1
                r -= dr
                c -= dc
            # Evaluate the score based on the count
            score += count * count

        return score


    # Count consecutive pieces in the given position
    def count_consecutive(self, board, row, col):
        count = 0
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        for dr, dc in directions:
            r, c = row, col
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == board[row][col]:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == board[row][col]:
                count += 1
                r -= dr
                c -= dc
        return count

class MinimaxAI(AI):
    def __init__(self, symbol, ai_agent_symbol, max_depth):
        super().__init__(symbol)
        self.ai_agent_symbol = ai_agent_symbol
        self.max_depth = max_depth
        self.first_turn = False # Plays a random first move if it starts the game
        self.random_moves = 0
        if symbol == PLAYER1:
            self.first_turn = True
            
    # Get the best move using the minimax algorithm
    def get_move(self, board):
        x = 2 # Number of random moves to play
        start_time = time.time()  # Record start time
        if (self.first_turn and self.random_moves < x): 
            valid_moves = [col for col in range(COLS) if BoardUtils.is_valid_move(board, col)]
            self.random_moves +=1
            print("RAND")
            return random.choice(valid_moves)
            
        # Check for winning moves
        for move in range(COLS):
            if BoardUtils.is_valid_move(board, move):
                row = BoardUtils.get_next_open_row(board, move)
                board[row][move] = self.symbol
                if BoardUtils.check_winner(board) == self.symbol:
                    board[row][move] = EMPTY
                    end_time = time.time()  # Record end time
                    print(f"Minimax AI move time: {end_time - start_time:.6f} seconds")
                    return move
                board[row][move] = EMPTY

        # Check for opponent's winning moves and block them
        opponent_symbol = PLAYER1 if self.ai_agent_symbol == PLAYER2 else PLAYER2
        for move in range(COLS):
            if BoardUtils.is_valid_move(board, move):
                row = BoardUtils.get_next_open_row(board, move)
                board[row][move] = opponent_symbol
                if BoardUtils.check_winner(board) == opponent_symbol:
                    board[row][move] = EMPTY
                    end_time = time.time()  # Record end time
                    print(f"Minimax AI move time: {end_time - start_time:.6f} seconds")
                    return move
                board[row][move] = EMPTY

        # If no immediate win or block, use minimax algorithm, alpha-beta pruning
        best_score = float('-inf')
        best_move = None

        for move in range(COLS):
            if BoardUtils.is_valid_move(board, move):
                row = BoardUtils.get_next_open_row(board, move)
                board[row][move] = self.ai_agent_symbol
                # Call the minimax algorithm to find the best move
                score = self.minimax(board, 0, False, float('-inf'), float('inf'))
                board[row][move] = EMPTY

                if score > best_score:
                    best_score = score
                    best_move = move

        end_time = time.time()  # Record end time
        print(f"Minimax AI move time: {end_time - start_time:.6f} seconds")
        return best_move

    # Minimax algorithm for decision-making
    def minimax(self, board, depth, is_maximizing_player, alpha, beta):
        # Check if the game has reached a terminal state or the maximum depth
        if depth == self.max_depth or BoardUtils.check_winner(board):
            return self.evaluate_state(board)

        if is_maximizing_player:
            max_score = float('-inf')

            # Explore possible moves for the maximizing player
            for move in range(COLS):
                if BoardUtils.is_valid_move(board, move):
                    row = BoardUtils.get_next_open_row(board, move)
                    board[row][move] = self.ai_agent_symbol
                    # Recursively call minimax for the next state
                    score = self.minimax(board, depth + 1, False, alpha, beta)
                    board[row][move] = EMPTY

                    max_score = max(max_score, score)
                    alpha = max(alpha, max_score)
                    if alpha >= beta:
                        break

            return max_score
        else:
            min_score = float('inf')

            # Explore possible moves for the minimizing player
            for move in range(COLS):
                if BoardUtils.is_valid_move(board, move):
                    row = BoardUtils.get_next_open_row(board, move)
                    board[row][move] = PLAYER1 if self.ai_agent_symbol == PLAYER2 else PLAYER2
                    # Recursively call minimax for the next state
                    score = self.minimax(board, depth + 1, True, alpha, beta)
                    board[row][move] = EMPTY

                    min_score = min(min_score, score)
                    beta = min(beta, min_score)
                    if beta <= alpha:
                        break

            return min_score

    # Evaluate the state of the game for the AI
    def evaluate_state(self, board):
        winner = BoardUtils.check_winner(board)
        ai_symbol = self.ai_agent_symbol
        opponent_symbol = PLAYER1 if ai_symbol == PLAYER2 else PLAYER2
        if winner == ai_symbol:
            return 100
        elif winner == opponent_symbol:
            return -100
        elif all(board[0][col] != EMPTY for col in range(COLS)):  # All positions are filled
            return 0  # Game is a draw
        else:
            ai_score = self.evaluate_position(board, ai_symbol)
            opponent_score = self.evaluate_position(board, opponent_symbol)
            return ai_score - opponent_score

    # Evaluate the position of pieces on the board for a player
    def evaluate_position(self, board, player):
        score = 0
        # Evaluate rows
        for r in range(ROWS):
            for c in range(COLS - 3):
                score += self.evaluate_window(board[r][c:c+4], player)
        # Evaluate columns
        for c in range(COLS):
            for r in range(ROWS - 3):
                score += self.evaluate_window([board[i][c] for i in range(r, r+4)], player)
        # Evaluate positive diagonals
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                score += self.evaluate_window([board[r+i][c+i] for i in range(4)], player)
        # Evaluate negative diagonals
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                score += self.evaluate_window([board[r+3-i][c+i] for i in range(4)], player)
        return score

    # Evaluate a window of positions on the board
    def evaluate_window(self, window, player):
        opp_player = PLAYER1 if player == PLAYER2 else PLAYER2
        score = 0
        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(player) == 2 and window.count(EMPTY) == 2:
            score += 2
        if window.count(opp_player) == 3 and window.count(EMPTY) == 1:
            score -= 4
        return score


# Class representing the Connect4 game
class Connect4:
    def __init__(self):
        # Initialize the game window
        self.root = tk.Tk()
        self.root.title("Connect 4")
        self.canvas = tk.Canvas(self.root, width=700, height=600, bg="white")
        self.canvas.pack()
        self.board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        self.turn = PLAYER1
        self.draw_board()
        self.canvas.bind("<Button-1>", self.drop_piece)
        self.ai_opponent = None # AI opponent 1
        self.ai_opponent2 = None # AI opponent 2
        self.setup_ui()
        self.root.mainloop()

    # Set up user interface for selecting game mode
    def setup_ui(self):
        self.label = tk.Label(self.root, text="Choose Mode:")
        self.label.pack()
        self.ai_options = tk.StringVar(self.root)
        self.ai_options.set("Human vs Minimax")
        self.dropdown = tk.OptionMenu(self.root, self.ai_options, "Human vs Minimax", "Minimax vs Human", "Minimax vs Minimax", "Random AI", "Greedy AI","Heuristic AI",
                                      "Greedy vs Minimax","Greedy vs Heuristic",
                                      "Heuristic vs Greedy", "Heuristic vs Minimax",
                                      "Minimax vs Greedy", "Minimax vs Heuristic",)
        self.dropdown.pack()
        self.start_button = tk.Button(self.root, text="Start Game", command=self.start_game)
        self.start_button.pack()

    # Start the game based on the selected mode
    def start_game(self):
        ai_option = self.ai_options.get()
        if ai_option == "Human vs Minimax":
            self.ai_opponent = MinimaxAI(symbol=PLAYER2, ai_agent_symbol=2, max_depth = 5) # Change depth as you wish
        elif ai_option == "Random AI":
            self.ai_opponent = RandomAI(PLAYER2)
        elif ai_option == "Greedy AI":
            self.ai_opponent = GreedyAI(PLAYER2)
        elif ai_option == "Heuristic AI":
            self.ai_opponent = HeuristicAI(PLAYER2)
        elif ai_option == "Greedy vs Minimax":
            self.ai_opponent = GreedyAI(PLAYER1)
            self.ai_opponent2 = MinimaxAI(symbol=PLAYER2, ai_agent_symbol=2, max_depth = 1)
        elif ai_option == "Greedy vs Heuristic":
            self.ai_opponent = GreedyAI(PLAYER1)
            self.ai_opponent2 = HeuristicAI(PLAYER2)
        elif ai_option == "Heuristic vs Minimax":
            self.ai_opponent = HeuristicAI(PLAYER1)
            self.ai_opponent2 = MinimaxAI(symbol=PLAYER2, ai_agent_symbol=2, max_depth = 1)
        elif ai_option == "Minimax vs Heuristic":
            self.ai_opponent = MinimaxAI(symbol=PLAYER1, ai_agent_symbol=1, max_depth = 1)
            self.ai_opponent2 = HeuristicAI(PLAYER1)
        elif ai_option == "Heuristic vs Greedy":
            self.ai_opponent = HeuristicAI(PLAYER1)
            self.ai_opponent2 = GreedyAI(PLAYER2)
        elif ai_option == "Minimax vs Greedy":
            self.ai_opponent = MinimaxAI(symbol=PLAYER1, ai_agent_symbol=1, max_depth = 1)
            self.ai_opponent2 = GreedyAI(PLAYER2)
        elif ai_option == "Minimax vs Minimax":
            self.ai_opponent = MinimaxAI(symbol=PLAYER1, ai_agent_symbol=1, max_depth = 4)
            self.ai_opponent2 = MinimaxAI(symbol=PLAYER2, ai_agent_symbol=2, max_depth = 4)
        elif ai_option == "Minimax vs Human":
            self.ai_opponent = MinimaxAI(symbol=PLAYER1, ai_agent_symbol=1, max_depth = 5)

        # Destroy UI elements after game starts
        self.start_button.destroy()
        self.label.destroy()
        self.dropdown.destroy()


        # Start the game based on selected mode
        if ai_option == "Greedy vs Minimax":
            self.greedy_vs_minimax_game()
        if ai_option == "Greedy vs Heuristic":
            self.greedy_vs_heuristic_game()
        if ai_option == "Heuristic vs Greedy":
            self.heuristic_vs_greedy_game()
        if ai_option == "Heuristic vs Minimax":
            self.heuristic_vs_minimax_game()
        if ai_option == "Minimax vs Greedy":
            self.minimax_vs_greedy_game()
        if ai_option == "Minimax vs Heuristic":
            self.minimax_vs_heuristic_game()
        if ai_option == "Minimax vs Minimax":
            self.minimax_vs_minimax_game()
        if ai_option == "Minimax vs Human":
            self.minimax_vs_human_game()

    # Handle game when it's Minimax AI vs Human
    def minimax_vs_human_game(self):
        while True:
            # Minimax AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Minimax MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Minimax"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        break

            # Human's turn
            if self.turn == PLAYER1:
                self.root.update()  # Update the GUI to allow human input
                self.root.wait_variable(self.turn)
                self.root.unbind("<Button-1>")  # Unbind the mouse click event


    # Handle game when it's Greedy vs Minimax AI
    def greedy_vs_minimax_game(self):
        while True:
            # Greedy AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER1
                        self.draw_board()
                        print("Greedy MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Greedy"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER2
                        self.root.update()
                        time.sleep(1)
                        break
            # Minimax AI's turn
            ai_col = self.ai_opponent2.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Minimax MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Minimax"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        time.sleep(1)  
                        break

    def minimax_vs_greedy_game(self):
        while True:
            # Minimax AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER1
                        self.draw_board()
                        print("Minimax MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Minimax"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER2
                        self.root.update()
                        time.sleep(1)  
                        break
            # Greedy AI's turn
            ai_col = self.ai_opponent2.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Greedy MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Greedy"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        time.sleep(1) 
                        break
                    
    def minimax_vs_heuristic_game(self):
        while True:
            # Minimax AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER1
                        self.draw_board()
                        print("Minimax MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Minimax"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER2
                        self.root.update()
                        time.sleep(1)  
                        break
            # Heuristic AI's turn
            ai_col = self.ai_opponent2.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Greedy MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Greedy"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        time.sleep(1) 
                        break
                    
    def minimax_vs_minimax_game(self):
        while True:
            # Minimax AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER1
                        self.draw_board()
                        print("Minimax MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Minimax Red"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER2
                        self.root.update()
                        time.sleep(1) 
                        break
            # Minimax AI's turn
            ai_col = self.ai_opponent2.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Minimax MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Minimax Yellow"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        time.sleep(1)
                        break
                                       
    # Handle game when it's Greedy vs Heuristic AI
    def greedy_vs_heuristic_game(self):
        while True:
            # Greedy AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER1
                        self.draw_board()
                        print("Greedy MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Greedy"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER2
                        self.root.update()
                        time.sleep(1)
                        break
            # Heuristic AI's turn
            ai_col = self.ai_opponent2.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Heuristic MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Heuristic"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        time.sleep(1)
                        break

    # Handle game when it's Heuristic vs Minimax AI 
    def heuristic_vs_minimax_game(self):
        while True:
            # Heuristic AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER1
                        self.draw_board()
                        print("Heuristic MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Heuristic"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER2
                        self.root.update()
                        time.sleep(1)
                        break
            # Minimax AI's turn
            ai_col = self.ai_opponent2.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Minimax MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Minimax"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        time.sleep(1)
                        break

    # Handle game when it's Heuristic vs Greedy AI
    def heuristic_vs_greedy_game(self):
        while True:
            # Heuristic AI's turn
            ai_col = self.ai_opponent.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER1
                        self.draw_board()
                        print("Heuristic MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Heuristic"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER2
                        self.root.update()
                        time.sleep(1)
                        break
            # Greedy AI's turn
            ai_col = self.ai_opponent2.get_move(self.board)
            if ai_col is not None:
                for row_ai in range(ROWS - 1, -1, -1):
                    if self.board[row_ai][ai_col] == EMPTY:
                        self.board[row_ai][ai_col] = PLAYER2
                        self.draw_board()
                        print("Greedy MOVED")
                        if self.check_win(row_ai, ai_col):
                            winner = "Greedy"
                            messagebox.showinfo("Winner", f"{winner} wins!")
                            return
                        self.turn = PLAYER1
                        self.root.update()
                        time.sleep(1)
                        break

    # Draw the game board
    def draw_board(self):
        self.canvas.delete("pieces")
        for row in range(ROWS):
            for col in range(COLS):
                x1 = col * 100
                y1 = row * 100
                x2 = x1 + 100
                y2 = y1 + 100
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black", tags="pieces")
                if self.board[row][col] == PLAYER1:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="red", outline="black", tags="pieces")
                elif self.board[row][col] == PLAYER2:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="yellow", outline="black", tags="pieces")

    # Handle dropping a piece on the board by the player
    def drop_piece(self, event):
        if self.ai_opponent is None:
            messagebox.showerror("Error", "Please select a mode opponent first.")
            return

        col = event.x // 100 # Get move from human  
        for row in range(ROWS - 1, -1, -1): # Iterate over rows
            if self.board[row][col] == EMPTY: # Check if cell is empty
                self.board[row][col] = self.turn # Place player piece
                self.draw_board() # Update board
                if self.check_win(row, col): # Check for winner
                    winner = "Red" if self.turn == PLAYER1 else "Yellow"
                    messagebox.showinfo("Winner", f"{winner} wins!")
                    return
                # Switch turns if no winner
                if self.turn == PLAYER1:
                    self.turn = PLAYER2
                    ai_col = self.ai_opponent.get_move(self.board) # Get move from ai
                    if ai_col is not None:
                        for row_ai in range(ROWS - 1, -1, -1):
                            if self.board[row_ai][ai_col] == EMPTY:
                                self.board[row_ai][ai_col] = PLAYER2
                                self.draw_board()
                                if self.check_win(row_ai, ai_col):
                                    winner = "Yellow"
                                    messagebox.showinfo("Winner", f"{winner} wins!")
                                    return
                                self.turn = PLAYER1
                                return

                elif self.turn == PLAYER2:
                    self.turn = PLAYER1
                return

    # Check if there's a winner after placing a piece
    def check_win(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]  # vertical, horizontal, diagonal (both ways)
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r = row + dr * i
                c = col + dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == self.turn:
                    count += 1
                else:
                    break
            for i in range(1, 4):
                r = row - dr * i
                c = col - dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == self.turn:
                    count += 1
                else:
                    break
            if count >= 4:
                return True

        # Check for a draw
        if all(self.board[0][col] != EMPTY for col in range(COLS)):
            messagebox.showinfo("Draw", "It's a draw!")
            self.root.quit()
            return False
            
        return False

# Main
if __name__ == "__main__":
    Connect4()
