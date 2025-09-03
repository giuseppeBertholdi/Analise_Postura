#função que imprime todos os números pares até n
def imprime_pares(n):
    for i in range(n+1):
        if i % 2 == 0:
            print(i)
imprime_pares(int(input("Digite um número: ")))

# #verificar se um número é primo
def primo(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

for n in range(1, 101):
    if primo(n):
        print(f"{n} é primo")

def hanoi(n, origem, destino, auxiliar):
    if n == 1:
        print(f"Mova o disco 1 de {origem} para {destino}")
        return
    hanoi(n-1, origem, auxiliar, destino)
    print(f"Mova o disco {n} de {origem} para {destino}")
    hanoi(n-1, auxiliar, destino, origem)

# Chamada da função para 3 discos
hanoi(3, 'A', 'C', 'B')


