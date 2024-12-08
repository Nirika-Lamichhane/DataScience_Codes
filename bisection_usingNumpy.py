# implement bisection method in python using numpy

import numpy as np

# define function
def f(x):
    return np.exp(x)-3*x

def bisection_method():
    print("\n Python program for bisection method\n")

    while True:
        try:
            x0=float(input("enter initial guess: "))
            x1=float(input("enter final guess: "))

            f0=f(x0)
            f1=f(x1)

            if f0*f1>0.0:
                print("wrong guess. the fucntion must have the negative sign")
                continue

            e=float(input("Enter the value of the predefined error (e):"))
            break
        except ValueError:
            print("invalid input. please enter numeric.")

    print("\n________________________________________________________________")
    print("Iterations \t x0 \t\t x1 \t\t x2 \t\t f(x2)")
    print("________________________________________________________________")

    iteration=1
    while True:
        x2=(x0+x1)/2
        f2=f(x2)

        print(f"{iteration:10d} \t {x0:.6f} \t {x1:.6f} \t {x2:.6f} \t {f2:.6e}")

        if 



    

            