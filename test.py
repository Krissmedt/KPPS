class test:
    methods = []
    def __init__(self):
        self.methods = [self.method1,self.method2,self.method3]
    
    def method1(self,x):
        x += 1
        return x
    
    def method2(self,x):
        x += 2
        return x
    
    def method3(self,x):
        x += 3
        return x
    
    def run(self,x):
        for method in self.methods:
            x = method(x)
    
        return x
    
test1 = test()
print(test1.run(1))