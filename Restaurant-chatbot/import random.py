import random

def solution(A):
    k = 0
    m = 1
    N = random.randint(1,100000)
    print(N)
    while k <= N:
        b = random.randint(-1000000,1000000)
        A[k] = b
        k = k + 1
    while k<=N:
        if m == A[k]:
            m = m + 1
        k = k + 1
    print(m)

if __name__ == "__main__":
    A = []
    solution(A)