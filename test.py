import hotspot

method = 'multiquadric'

for i in range(20, 70, 5):
    print(method)
    
    hotspot.test(i, hotspot.funcEasy, -1, 1, -1, 1, method)
#    hotspot.test(i, hotspot.funcHard, -1, 1, -1, 1, method)
#    hotspot.test(i, hotspot.funcHard2, -1, 1, -1, 1, method)
    hotspot.test(i, hotspot.funcSound, -1, 1, -1, 1, method, 10) 
    hotspot.test(i, hotspot.funcSoundAtt, -1, 1, -1, 1, method, 10, -5) 
    hotspot.test(i, hotspot.funcSin, -1, 1, -1, 1, method)

'''
methods = ['cubic', 'quintic']
for i in range(2, 10):
    for method in methods:
        print(method)
        test(i, funcEasy, -1, 1, -1, 1, method)
        test(i, funcHard, -1, 1, -1, 1, method)
        test(i, funcHard2, -1, 1, -1, 1, method)
        test(i, funcSound, -1, 1, -1, 1, method, 10)
        test(i, funcSoundAtt, -1, 1, -1, 1, method, 10, -5)
        test(i, funcSin, -1, 1, -1, 1, method)
'''
