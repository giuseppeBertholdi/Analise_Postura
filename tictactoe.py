import tkinter as tk
from tkinter import messagebox

board = [[" "]*3 for _ in range(3)]
buttons = [[None]*3 for _ in range(3)]
current_player = "X"  # por padrão humano começa

def check_winner(board, player):
    for i in range(3):
        if all([cell == player for cell in board[i]]):
            return True
        if all([board[j][i] == player for j in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2-i] == player for i in range(3)]):
        return True
    return False

def is_draw(board):
    return all([cell != ' ' for row in board for cell in row])

def minimax(board, depth, is_maximizing):
    if check_winner(board, "O"):
        return 1
    if check_winner(board, "X"):
        return -1
    if is_draw(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == " ":
                    board[r][c] = "O"
                    score = minimax(board, depth + 1, False)
                    board[r][c] = " "
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == " ":
                    board[r][c] = "X"
                    score = minimax(board, depth + 1, True)
                    board[r][c] = " "
                    best_score = min(score, best_score)
        return best_score

def ai_turn():
    global current_player
    best_score = -float('inf')
    move = None
    for r in range(3):
        for c in range(3):
            if board[r][c] == " ":
                board[r][c] = "O"
                score = minimax(board, 0, False)
                board[r][c] = " "
                if score > best_score:
                    best_score = score
                    move = (r, c)
    if move:
        board[move[0]][move[1]] = "O"
        buttons[move[0]][move[1]]["text"] = "O"

    if check_winner(board, "O"):
        messagebox.showinfo("Fim de jogo", "IA ganhou!")
    elif is_draw(board):
        messagebox.showinfo("Fim de jogo", "Empate!")
    else:
        current_player = "X"  # passa a vez para o humano

def player_click(row, col):
    global current_player
    if current_player == "X" and board[row][col] == " ":
        board[row][col] = "X"
        buttons[row][col]["text"] = "X"
        if check_winner(board, "X"):
            messagebox.showinfo("Fim de jogo", "Você ganhou!")
        elif is_draw(board):
            messagebox.showinfo("Fim de jogo", "Empate!")
        else:
            current_player = "O"
            ai_turn()

def reset_game():
    global board, current_player
    board = [[" "]*3 for _ in range(3)]
    current_player = "X"
    for r in range(3):
        for c in range(3):
            buttons[r][c]["text"] = " "

def ai_start():
    global current_player
    reset_game()         # começa um novo jogo
    current_player = "O" # IA joga primeiro
    ai_turn()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Jogo da Velha")

# Botões do tabuleiro
for r in range(3):
    for c in range(3):
        button = tk.Button(root, text=" ", font=("Arial", 40), width=5, height=2,
                           command=lambda row=r, col=c: player_click(row, col))
        button.grid(row=r, column=c)
        buttons[r][c] = button

# Botão de reiniciar
restart_button = tk.Button(root, text="Reiniciar", font=("Arial", 20), command=reset_game)
restart_button.grid(row=3, column=0, columnspan=3)

# Botão para a IA começar
ai_start_button = tk.Button(root, text="IA Começa", font=("Arial", 20), command=ai_start)
ai_start_button.grid(row=4, column=0, columnspan=3)

root.mainloop()
