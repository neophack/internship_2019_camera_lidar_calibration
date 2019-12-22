

if __name__ == '__main__':
    from engine import Engine, PointProcessing, PointReader
    print('working')
    eng = Engine('C:\\Users\Tatiana\\Documents\\Python Scripts\\Camera+Lidar\\Python_class')
    print(eng)
    print(eng.path)

    eng.power(0)