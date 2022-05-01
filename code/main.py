from tensorflow import exp
from min_function import min_function

def main():
    f = lambda x, y: -exp(-((x/2)**2 + 5 * (y/2)**2)) + (x/2)**2 + 0.5 * (y/2)**2
    mm = min_function(f, [3., 3.], [0., 0.], 0.1)
    mm.plot2d()
    mm.plot3d(60, -70)
    mm.create_gif()    

if __name__ == "__main__":
    main()
